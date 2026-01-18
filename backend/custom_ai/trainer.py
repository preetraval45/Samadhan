import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import time


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [0] * padding_length

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)



class CustomTrainer:
    def __init__(self, model, tokenizer, device='cuda', learning_rate=3e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(self.device)

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': 0
        }


    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc='Training')

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids)

            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss


    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for input_ids, target_ids in dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits = self.model(input_ids)

                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss


    def train(self, train_texts, val_texts=None, epochs=10, batch_size=8, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)

        train_dataset = TextDataset(train_texts, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_texts:
            val_dataset = TextDataset(val_texts, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')

            train_loss = self.train_epoch(train_loader)
            self.training_stats['train_losses'].append(train_loss)

            print(f'Train Loss: {train_loss:.4f}')

            if val_loader:
                val_loss = self.validate(val_loader)
                self.training_stats['val_losses'].append(val_loss)
                print(f'Val Loss: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))
                    print('Saved best model')

            self.training_stats['epochs'] += 1

            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path)

        final_path = os.path.join(save_dir, 'final_model.pt')
        self.save_checkpoint(final_path)
        print(f'\nTraining complete. Final model saved to {final_path}')


    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
            }
        }

        torch.save(checkpoint, filepath)


    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

        print(f'Loaded checkpoint from {filepath}')
        print(f'Trained for {self.training_stats["epochs"]} epochs')



class DataPreprocessor:
    def __init__(self):
        pass


    def load_text_files(self, file_paths):
        texts = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
        return texts


    def load_json_dataset(self, file_path, text_field='text'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        if isinstance(data, list):
            for item in data:
                if text_field in item:
                    texts.append(item[text_field])
        elif isinstance(data, dict):
            if text_field in data:
                texts.append(data[text_field])

        return texts


    def split_dataset(self, texts, val_split=0.1, test_split=0.1):
        np.random.shuffle(texts)

        total_size = len(texts)
        val_size = int(total_size * val_split)
        test_size = int(total_size * test_split)
        train_size = total_size - val_size - test_size

        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size+val_size]
        test_texts = texts[train_size+val_size:]

        return train_texts, val_texts, test_texts


    def clean_text(self, text):
        text = text.strip()
        text = ' '.join(text.split())
        return text
