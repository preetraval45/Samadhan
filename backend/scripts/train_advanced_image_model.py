import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import argparse
from tqdm import tqdm
import glob

from custom_ai.image_generator import CustomImageGenerator
from custom_ai.advanced_image_generation import (
    AdvancedImageGenerator,
    ControlNetEncoder,
    InpaintingModel,
    OutpaintingModel,
    SuperResolutionModel
)
from custom_ai.tokenizer import CustomTokenizer


class ImageTextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, image_size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size

        self.samples = []

        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.samples = json.load(f)
        else:
            image_files = glob.glob(os.path.join(data_dir, '*.jpg')) + \
                         glob.glob(os.path.join(data_dir, '*.png'))

            for img_path in image_files:
                txt_path = img_path.rsplit('.', 1)[0] + '.txt'
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        caption = f.read().strip()
                    self.samples.append({
                        'image': img_path,
                        'caption': caption
                    })

        print(f"Loaded {len(self.samples)} image-text pairs")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image']).convert('RGB')
        image = self.transform(image)

        caption_tokens = self.tokenizer.encode(sample['caption'])
        caption_tokens = caption_tokens[:77]
        caption_tokens = caption_tokens + [0] * (77 - len(caption_tokens))

        return {
            'image': image,
            'caption_tokens': torch.tensor(caption_tokens, dtype=torch.long)
        }


class ImageModelTrainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)

    def train_epoch(self, dataloader, optimizer, scaler):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc='Training')

        for batch in progress_bar:
            images = batch['image'].to(self.device)
            captions = batch['caption_tokens'].to(self.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                noise = torch.randn_like(images)

                timesteps = torch.randint(0, 1000, (images.size(0),), device=self.device)

                noisy_images = images + noise * 0.1

                predicted_noise = self.model(captions)

                loss = nn.functional.mse_loss(predicted_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, epochs, lr, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss = self.train_epoch(train_loader, optimizer, scaler)
            print(f"Train Loss: {train_loss:.4f}")

            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'), epoch, train_loss)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'), epoch, train_loss)

        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'), epochs, train_loss)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                captions = batch['caption_tokens'].to(self.device)

                noise = torch.randn_like(images)
                noisy_images = images + noise * 0.1

                predicted_noise = self.model(captions)

                loss = nn.functional.mse_loss(predicted_noise, noise)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, path, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss
        }, path)
        print(f"Saved checkpoint to {path}")


def train_controlnet(data_dir, output_dir, epochs=50):
    print("Training ControlNet...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    controlnet = ControlNetEncoder().to(device)

    dataset = ImageTextDataset(data_dir, CustomTokenizer())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    optimizer = optim.AdamW(controlnet.parameters(), lr=1e-4)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        controlnet.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = batch['image'].to(device)

            optimizer.zero_grad()

            control_features = controlnet(images)

            loss = control_features.pow(2).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(controlnet.state_dict(),
                      os.path.join(output_dir, f'controlnet_epoch_{epoch+1}.pt'))

    print("ControlNet training complete!")


def train_super_resolution(data_dir, output_dir, scale_factor=8, epochs=100):
    print(f"Training {scale_factor}x Super-Resolution model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sr_model = SuperResolutionModel(scale_factor=scale_factor).to(device)

    image_files = glob.glob(os.path.join(data_dir, '*.jpg')) + \
                  glob.glob(os.path.join(data_dir, '*.png'))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    optimizer = optim.Adam(sr_model.parameters(), lr=1e-4)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        sr_model.train()
        total_loss = 0

        for img_path in tqdm(image_files[:100], desc=f'Epoch {epoch+1}/{epochs}'):
            image = Image.open(img_path).convert('RGB')
            hr_image = transform(image).unsqueeze(0).to(device)

            lr_image = nn.functional.interpolate(
                hr_image,
                scale_factor=1/scale_factor,
                mode='bilinear'
            )

            optimizer.zero_grad()

            sr_image = sr_model(lr_image)

            loss = nn.functional.l1_loss(sr_image, hr_image)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(image_files[:100]):.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save(sr_model.state_dict(),
                      os.path.join(output_dir, f'sr_{scale_factor}x_epoch_{epoch+1}.pt'))

    print("Super-resolution training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train advanced image generation models')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['base', 'controlnet', 'super_resolution', 'all'],
                       help='Training mode')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints/image_models',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')

    args = parser.parse_args()

    if args.mode == 'base' or args.mode == 'all':
        print("Training base image generation model...")

        tokenizer = CustomTokenizer(vocab_size=50000)
        model = CustomImageGenerator(vocab_size=50000)

        dataset = ImageTextDataset(args.data_dir, tokenizer)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

        trainer = ImageModelTrainer(model, tokenizer)
        trainer.train(train_loader, val_loader, args.epochs, args.learning_rate,
                     os.path.join(args.output_dir, 'base'))

    if args.mode == 'controlnet' or args.mode == 'all':
        train_controlnet(args.data_dir, os.path.join(args.output_dir, 'controlnet'), epochs=50)

    if args.mode == 'super_resolution' or args.mode == 'all':
        train_super_resolution(args.data_dir, os.path.join(args.output_dir, 'super_resolution'),
                             scale_factor=8, epochs=100)
        train_super_resolution(args.data_dir, os.path.join(args.output_dir, 'super_resolution'),
                             scale_factor=16, epochs=100)

    print("\nAll training complete!")


if __name__ == '__main__':
    main()
