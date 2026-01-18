"""
Executable Training Script for Custom Image Generation Model
This trains a custom Stable Diffusion model from scratch or fine-tunes existing one
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional
import wandb

class ImageTextDataset(Dataset):
    """Dataset for training image generation models"""

    def __init__(self, image_dir: str, caption_file: str, resolution: int = 512):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.image_files = []
        self.captions = {}

        # Load captions
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    filename, caption = parts
                    self.captions[filename] = caption
                    if (self.image_dir / filename).exists():
                        self.image_files.append(filename)

        print(f"Loaded {len(self.image_files)} images with captions")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = self.image_dir / filename
        caption = self.captions[filename]

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        return {
            'pixel_values': image,
            'text': caption
        }


class CustomImageGenerationModel(nn.Module):
    """Custom Image Generation Model Architecture"""

    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    ):
        super().__init__()

        # Load pretrained components (FREE models)
        print(f"Loading base model: {pretrained_model_name}")

        # VAE Encoder/Decoder
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name,
            subfolder="vae"
        )

        # Text Encoder (CLIP)
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name,
            subfolder="text_encoder"
        )

        # U-Net Denoiser (this is what we'll train)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet"
        )

        # Noise scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name,
            subfolder="scheduler"
        )

        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name,
            subfolder="tokenizer"
        )

        # Freeze VAE and text encoder (only train U-Net)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        print("✓ Model components loaded")

    def forward(self, pixel_values, text):
        """Forward pass for training"""

        # Encode images to latent space
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample random timestep
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()

        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(latents.device)

        text_embeddings = self.text_encoder(text_inputs)[0]

        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

        # Calculate loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss


class ModelTrainer:
    """Trainer for custom image generation model"""

    def __init__(
        self,
        model: CustomImageGenerationModel,
        train_dataset: Dataset,
        output_dir: str = "./models/custom_image_gen",
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16",
        use_wandb: bool = True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision
        )

        # DataLoader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_dataloader)
        )

        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )

        # Initialize Weights & Biases for tracking
        if use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project="custom-image-generation",
                config={
                    "learning_rate": learning_rate,
                    "epochs": num_epochs,
                    "batch_size": batch_size
                }
            )

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Dataset size: {len(self.train_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total steps: {self.num_epochs * len(self.train_dataloader)}")

        global_step = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.accelerator.is_main_process
            )

            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    loss = self.model(
                        pixel_values=batch['pixel_values'],
                        text=batch['text']
                    )

                    # Backward pass
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.item()
                    global_step += 1

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                    })

                    # Log to wandb
                    if self.accelerator.is_main_process and global_step % 10 == 0:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": self.lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch
                        }, step=global_step)

            # Epoch complete
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

        # Save final model
        self.save_checkpoint("final")
        print("✅ Training complete!")

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = self.output_dir / f"checkpoint-{epoch}"
            checkpoint_dir.mkdir(exist_ok=True)

            # Unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            # Save U-Net (the trained component)
            unwrapped_model.unet.save_pretrained(checkpoint_dir / "unet")

            print(f"✓ Checkpoint saved to {checkpoint_dir}")


def main():
    """Main execution function"""

    # Configuration
    CONFIG = {
        "pretrained_model": "runwayml/stable-diffusion-v1-5",  # FREE base model
        "image_dir": "./datasets/training_images",  # Your image directory
        "caption_file": "./datasets/captions.txt",  # Image captions
        "output_dir": "./models/custom_image_gen",
        "resolution": 512,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "num_epochs": 100,
        "mixed_precision": "fp16",
        "use_wandb": True
    }

    print("=" * 60)
    print("CUSTOM IMAGE GENERATION MODEL TRAINING")
    print("=" * 60)
    print(f"Base Model: {CONFIG['pretrained_model']}")
    print(f"Resolution: {CONFIG['resolution']}x{CONFIG['resolution']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print("=" * 60)

    # Check if dataset exists
    if not os.path.exists(CONFIG['image_dir']):
        print(f"❌ Dataset directory not found: {CONFIG['image_dir']}")
        print("\nTo prepare your dataset:")
        print("1. Create a directory with your training images")
        print("2. Create captions.txt with format: filename.jpg<TAB>caption")
        print("\nExample captions.txt:")
        print("image1.jpg\tA beautiful sunset over mountains")
        print("image2.jpg\tA portrait of a person smiling")
        return

    # Create dataset
    dataset = ImageTextDataset(
        image_dir=CONFIG['image_dir'],
        caption_file=CONFIG['caption_file'],
        resolution=CONFIG['resolution']
    )

    # Create model
    model = CustomImageGenerationModel(
        pretrained_model_name=CONFIG['pretrained_model']
    )

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_dataset=dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        num_epochs=CONFIG['num_epochs'],
        mixed_precision=CONFIG['mixed_precision'],
        use_wandb=CONFIG['use_wandb']
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
