"""
Executable Training Script for Custom Deepfake Model
Trains a high-quality face swapping model from scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import insightface
from typing import Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceDataset(Dataset):
    """Dataset for training deepfake models"""

    def __init__(self, data_dir: str, resolution: int = 256):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.face_detector = insightface.app.FaceAnalysis()
        self.face_detector.prepare(ctx_id=-1)  # CPU

        # Find all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(list(self.data_dir.glob(f"**/{ext}")))

        print(f"Found {len(self.image_files)} images")

        # Augmentation pipeline
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect and align face
        faces = self.face_detector.get(image)

        if len(faces) == 0:
            # If no face detected, return random crop
            h, w = image.shape[:2]
            x = np.random.randint(0, max(1, w - self.resolution))
            y = np.random.randint(0, max(1, h - self.resolution))
            face_crop = image[y:y+self.resolution, x:x+self.resolution]
        else:
            # Extract aligned face
            face = faces[0]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Add padding
            padding = 30
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            face_crop = image[y1:y2, x1:x2]

        # Resize to target resolution
        face_crop = cv2.resize(face_crop, (self.resolution, self.resolution))

        # Apply augmentation
        augmented = self.transform(image=face_crop)
        face_tensor = augmented['image']

        return face_tensor


class EncoderBlock(nn.Module):
    """Encoder block for feature extraction"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x), x  # pooled + skip connection


class DecoderBlock(nn.Module):
    """Decoder block for reconstruction"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DeepfakeAutoencoder(nn.Module):
    """
    Custom Deepfake Model Architecture
    Based on autoencoder with shared encoder and separate decoders
    """

    def __init__(self, resolution: int = 256):
        super().__init__()
        self.resolution = resolution

        # Shared Encoder (feature extraction)
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # Decoder A (for person A)
        self.dec_a4 = DecoderBlock(1024, 512)
        self.dec_a3 = DecoderBlock(512, 256)
        self.dec_a2 = DecoderBlock(256, 128)
        self.dec_a1 = DecoderBlock(128, 64)
        self.out_a = nn.Conv2d(64, 3, 1)

        # Decoder B (for person B)
        self.dec_b4 = DecoderBlock(1024, 512)
        self.dec_b3 = DecoderBlock(512, 256)
        self.dec_b2 = DecoderBlock(256, 128)
        self.dec_b1 = DecoderBlock(128, 64)
        self.out_b = nn.Conv2d(64, 3, 1)

    def encode(self, x):
        """Encode image to latent features"""
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        bottleneck = self.bottleneck(x4)
        return bottleneck, (skip1, skip2, skip3, skip4)

    def decode_a(self, z, skips):
        """Decode with decoder A"""
        skip1, skip2, skip3, skip4 = skips
        x = self.dec_a4(z, skip4)
        x = self.dec_a3(x, skip3)
        x = self.dec_a2(x, skip2)
        x = self.dec_a1(x, skip1)
        return torch.tanh(self.out_a(x))

    def decode_b(self, z, skips):
        """Decode with decoder B"""
        skip1, skip2, skip3, skip4 = skips
        x = self.dec_b4(z, skip4)
        x = self.dec_b3(x, skip3)
        x = self.dec_b2(x, skip2)
        x = self.dec_b1(x, skip1)
        return torch.tanh(self.out_b(x))

    def forward(self, x, decoder='a'):
        """Forward pass"""
        z, skips = self.encode(x)
        if decoder == 'a':
            return self.decode_a(z, skips)
        else:
            return self.decode_b(z, skips)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.features(pred)
        target_features = self.features(target)
        return F.mse_loss(pred_features, target_features)


class DeepfakeTrainer:
    """Trainer for deepfake model"""

    def __init__(
        self,
        model: DeepfakeAutoencoder,
        dataset_a_path: str,
        dataset_b_path: str,
        output_dir: str = "./models/custom_deepfake",
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create datasets
        self.dataset_a = FaceDataset(dataset_a_path)
        self.dataset_b = FaceDataset(dataset_b_path)

        self.dataloader_a = DataLoader(
            self.dataset_a, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        self.dataloader_b = DataLoader(
            self.dataset_b, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )

        # Optimizers
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(device)

        self.num_epochs = num_epochs

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        dataloader = zip(self.dataloader_a, self.dataloader_b)
        num_batches = min(len(self.dataloader_a), len(self.dataloader_b))

        progress_bar = tqdm(
            dataloader,
            total=num_batches,
            desc=f"Epoch {epoch+1}/{self.num_epochs}"
        )

        for batch_a, batch_b in progress_bar:
            batch_a = batch_a.to(self.device)
            batch_b = batch_b.to(self.device)

            # Train on person A
            pred_a = self.model(batch_a, decoder='a')
            loss_a_recon = self.l1_loss(pred_a, batch_a)
            loss_a_perceptual = self.perceptual_loss(pred_a, batch_a)
            loss_a = loss_a_recon + 0.1 * loss_a_perceptual

            # Train on person B
            pred_b = self.model(batch_b, decoder='b')
            loss_b_recon = self.l1_loss(pred_b, batch_b)
            loss_b_perceptual = self.perceptual_loss(pred_b, batch_b)
            loss_b = loss_b_recon + 0.1 * loss_b_perceptual

            # Total loss
            loss = loss_a + loss_b

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'loss_a': f"{loss_a.item():.4f}",
                'loss_b': f"{loss_b.item():.4f}"
            })

        return total_loss / num_batches

    def train(self):
        """Main training loop"""
        print(f"Training deepfake model for {self.num_epochs} epochs")
        print(f"Dataset A: {len(self.dataset_a)} images")
        print(f"Dataset B: {len(self.dataset_b)} images")

        for epoch in range(self.num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint("final")
        print("✅ Training complete!")

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"deepfake_model_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")


def main():
    """Main execution"""

    CONFIG = {
        "person_a_dataset": "./datasets/person_a",  # Folder with person A images
        "person_b_dataset": "./datasets/person_b",  # Folder with person B images
        "output_dir": "./models/custom_deepfake",
        "resolution": 256,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print("=" * 60)
    print("CUSTOM DEEPFAKE MODEL TRAINING")
    print("=" * 60)
    print(f"Resolution: {CONFIG['resolution']}x{CONFIG['resolution']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Device: {CONFIG['device']}")
    print("=" * 60)

    # Create model
    model = DeepfakeAutoencoder(resolution=CONFIG['resolution'])

    # Create trainer
    trainer = DeepfakeTrainer(
        model=model,
        dataset_a_path=CONFIG['person_a_dataset'],
        dataset_b_path=CONFIG['person_b_dataset'],
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        num_epochs=CONFIG['num_epochs'],
        device=CONFIG['device']
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
