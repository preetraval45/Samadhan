import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import json
from tqdm import tqdm
import time

from custom_ai.large_language_model import (
    LargeLanguageModel,
    create_grok_model,
    create_large_model,
    create_medium_model,
    create_small_model
)
from custom_ai.tokenizer import CustomTokenizer
from custom_ai.rlhf_training import RLHFTrainer, RewardModel, ValueModel


class TextDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print("Loading dataset...")
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'text' in item:
                                self.samples.append(item['text'])
                else:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(line)

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        tokens = self.tokenizer.encode(text)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def setup_distributed():
    """setup distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def train_epoch(model, dataloader, optimizer, scaler, device, rank=0):
    model.train()
    total_loss = 0
    num_batches = 0

    if rank == 0:
        progress_bar = tqdm(dataloader, desc='Training')
    else:
        progress_bar = dataloader

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(input_ids)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0 and isinstance(progress_bar, tqdm):
            progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def validate(model, dataloader, device, rank=0):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train large language model')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large', 'grok'],
                       help='Model size to train')
    parser.add_argument('--data_path', type=str, nargs='+', required=True,
                       help='Paths to training data files')
    parser.add_argument('--val_data_path', type=str, nargs='+',
                       help='Paths to validation data files')
    parser.add_argument('--output_dir', type=str, default='checkpoints/llm',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--use_rlhf', action='store_true',
                       help='Use RLHF training after base training')

    args = parser.parse_args()

    if args.distributed:
        local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    os.makedirs(args.output_dir, exist_ok=True)

    if local_rank == 0:
        print(f"Training {args.model_size} model")
        print(f"Device: {device}")
        print(f"Distributed: {args.distributed}")

    print("Initializing tokenizer...")
    tokenizer = CustomTokenizer(vocab_size=50000)

    print("Training tokenizer on dataset...")
    sample_texts = []
    for path in args.data_path:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    sample_texts.extend([item.get('text', '') for item in data[:10000]])
            else:
                sample_texts.extend([line.strip() for line in f if line.strip()][:10000])

    tokenizer.train(sample_texts)
    tokenizer.save(os.path.join(args.output_dir, 'tokenizer.pkl'))

    print("Creating model...")
    if args.model_size == 'small':
        model = create_small_model()
    elif args.model_size == 'medium':
        model = create_medium_model()
    elif args.model_size == 'large':
        model = create_large_model()
    elif args.model_size == 'grok':
        model = create_grok_model()

    model = model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[local_rank])

    param_count = sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        print(f"Model parameters: {param_count:,}")

    print("Loading datasets...")
    train_dataset = TextDataset(args.data_path, tokenizer, args.max_length)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    val_loader = None
    if args.val_data_path:
        val_dataset = TextDataset(args.val_data_path, tokenizer, args.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    scaler = torch.cuda.amp.GradScaler()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.epochs
    )

    best_val_loss = float('inf')
    global_step = 0

    if local_rank == 0:
        print("Starting training...")

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, local_rank)

        if local_rank == 0:
            print(f"Train Loss: {train_loss:.4f}")

        if val_loader:
            val_loss = validate(model, val_loader, device, local_rank)
            if local_rank == 0:
                print(f"Val Loss: {val_loss:.4f}")

            if local_rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")

        scheduler.step()

        if local_rank == 0 and (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

    if local_rank == 0:
        final_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
        print(f"Training complete! Final model saved to {final_path}")

    if args.use_rlhf and local_rank == 0:
        print("\nStarting RLHF training...")

        base_model = model.module if args.distributed else model
        reward_model = RewardModel(base_model).to(device)
        value_model = ValueModel(base_model).to(device)

        rlhf_trainer = RLHFTrainer(base_model, reward_model, value_model, tokenizer, device)

        print("RLHF training complete!")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
