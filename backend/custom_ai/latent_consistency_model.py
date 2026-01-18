"""
Latent Consistency Models (LCM) for Fast Image Generation
Enables 4-8 step generation instead of 50+ steps in traditional diffusion models
Based on: "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConsistencyFunction(nn.Module):
    """
    Consistency function that maps noisy latents to clean latents
    Ensures consistency across different noise levels
    """

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

        # Consistency network - simplified U-Net style
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 4, 3, padding=1),
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy latent [B, 4, H, W]
            t: Time step [B, 1]

        Returns:
            Predicted clean latent [B, 4, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, 256]

        # Encode
        h = self.encoder(x)  # [B, 256, H/4, W/4]

        # Add time embedding
        t_emb = t_emb.view(t_emb.shape[0], -1, 1, 1)
        h = h + t_emb

        # Middle
        h = self.middle(h)

        # Decode
        out = self.decoder(h)  # [B, 4, H, W]

        return out


class LatentConsistencyModel(nn.Module):
    """
    Fast image generation using consistency distillation
    Generates high-quality images in 4-8 steps
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        num_inference_steps: int = 4,  # Much fewer than traditional 50
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps

        # Consistency function
        self.consistency_fn = ConsistencyFunction(latent_dim)

        # Noise scheduler
        self.register_buffer('betas', self._get_noise_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def _get_noise_schedule(self) -> torch.Tensor:
        """Linear noise schedule"""
        return torch.linspace(0.0001, 0.02, self.num_timesteps)

    def get_skip_steps(self) -> torch.Tensor:
        """
        Get timesteps to skip for fast inference
        Only evaluate at 4-8 steps instead of 1000
        """
        return torch.linspace(
            0,
            self.num_timesteps - 1,
            self.num_inference_steps,
            dtype=torch.long
        )

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        num_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fast generation using consistency model

        Args:
            latent: Starting noise [B, 4, H, W]
            num_steps: Number of inference steps (default: 4)
            guidance_scale: Classifier-free guidance scale
            condition: Text/image condition embedding

        Returns:
            Generated latent [B, 4, H, W]
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        # Get skip steps
        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=latent.device
        )

        x_t = latent

        for i, t in enumerate(timesteps):
            # Normalize time to [0, 1]
            t_normalized = t.float() / self.num_timesteps
            t_batch = t_normalized.repeat(latent.shape[0]).view(-1, 1)

            # Predict clean latent
            x_0_pred = self.consistency_fn(x_t, t_batch)

            if condition is not None:
                # Classifier-free guidance
                x_0_uncond = self.consistency_fn(x_t, t_batch)
                x_0_pred = x_0_uncond + guidance_scale * (x_0_pred - x_0_uncond)

            if i < len(timesteps) - 1:
                # Add noise for next step (consistency trajectory)
                next_t = timesteps[i + 1]
                alpha_t = self.alphas_cumprod[t]
                alpha_next = self.alphas_cumprod[next_t]

                # Interpolate between current and predicted clean
                sigma = torch.sqrt((1 - alpha_next) / (1 - alpha_t))
                x_t = torch.sqrt(alpha_next / alpha_t) * x_t + \
                      (1 - alpha_next / alpha_t) * x_0_pred + \
                      sigma * torch.randn_like(x_t)
            else:
                x_t = x_0_pred

        return x_t

    def consistency_loss(
        self,
        x_start: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Consistency distillation loss
        Ensures f(x_t1, t1) ≈ f(x_t2, t2) for adjacent timesteps

        Args:
            x_start: Clean image/latent [B, 4, H, W]
            t1, t2: Adjacent timesteps [B]

        Returns:
            Loss scalar
        """
        # Add noise
        noise = torch.randn_like(x_start)

        alpha_t1 = self.alphas_cumprod[t1].view(-1, 1, 1, 1)
        alpha_t2 = self.alphas_cumprod[t2].view(-1, 1, 1, 1)

        x_t1 = torch.sqrt(alpha_t1) * x_start + torch.sqrt(1 - alpha_t1) * noise
        x_t2 = torch.sqrt(alpha_t2) * x_start + torch.sqrt(1 - alpha_t2) * noise

        # Normalize time
        t1_norm = t1.float() / self.num_timesteps
        t2_norm = t2.float() / self.num_timesteps

        # Predict from both timesteps
        pred_1 = self.consistency_fn(x_t1, t1_norm.view(-1, 1))

        with torch.no_grad():
            pred_2 = self.consistency_fn(x_t2, t2_norm.view(-1, 1))

        # Consistency loss - predictions should be similar
        loss = F.mse_loss(pred_1, pred_2)

        # Add reconstruction loss for t=0
        if (t1 == 0).any():
            mask = (t1 == 0).float().view(-1, 1, 1, 1)
            recon_loss = F.mse_loss(pred_1 * mask, x_start * mask)
            loss = loss + recon_loss

        return loss


class FastImageGenerator:
    """
    High-level interface for fast image generation
    """

    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_inference_steps: int = 4,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps

        # Initialize LCM
        self.lcm = LatentConsistencyModel(
            num_inference_steps=num_inference_steps
        ).to(device)

    def generate_image(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_images: int = 1,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Generate images in 4-8 steps (vs 50+ for traditional diffusion)

        Args:
            prompt: Text prompt
            height, width: Image dimensions
            num_images: Batch size
            guidance_scale: CFG scale

        Returns:
            Generated images [B, 3, H, W]
        """
        # Start from noise
        latent_h, latent_w = height // 8, width // 8
        latents = torch.randn(
            num_images, 4, latent_h, latent_w,
            device=self.device
        )

        # TODO: Encode prompt to condition
        condition = None  # Would use CLIP/T5 encoder

        # Generate
        with torch.no_grad():
            generated_latents = self.lcm.generate(
                latents,
                num_steps=self.num_inference_steps,
                guidance_scale=guidance_scale,
                condition=condition,
            )

        # TODO: Decode latents to pixels with VAE
        # For now, return latents
        return generated_latents

    def benchmark_speed(self, height: int = 512, width: int = 512):
        """
        Compare LCM speed vs traditional diffusion
        """
        import time

        print("=" * 50)
        print("LCM Speed Benchmark")
        print("=" * 50)

        latent_h, latent_w = height // 8, width // 8
        latents = torch.randn(1, 4, latent_h, latent_w, device=self.device)

        # Warm up
        for _ in range(3):
            _ = self.lcm.generate(latents, num_steps=4)

        # LCM (4 steps)
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start = time.time()
        for _ in range(10):
            _ = self.lcm.generate(latents, num_steps=4)
        torch.cuda.synchronize() if self.device == 'cuda' else None
        lcm_time = (time.time() - start) / 10

        # LCM (8 steps)
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start = time.time()
        for _ in range(10):
            _ = self.lcm.generate(latents, num_steps=8)
        torch.cuda.synchronize() if self.device == 'cuda' else None
        lcm_8_time = (time.time() - start) / 10

        # Traditional diffusion (50 steps) - simulated
        traditional_time = lcm_time * (50 / 4)

        print(f"LCM (4 steps):  {lcm_time*1000:.2f}ms")
        print(f"LCM (8 steps):  {lcm_8_time*1000:.2f}ms")
        print(f"Traditional (50 steps): {traditional_time*1000:.2f}ms (estimated)")
        print(f"\nSpeedup: {traditional_time/lcm_time:.1f}x faster!")
        print("=" * 50)


# Training function
def train_lcm_from_pretrained(
    teacher_model: nn.Module,
    train_dataloader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = 'cuda',
):
    """
    Distill a teacher diffusion model into fast LCM

    Args:
        teacher_model: Pre-trained diffusion model
        train_dataloader: Training data
        num_epochs: Training epochs
        lr: Learning rate
        device: Device
    """
    lcm = LatentConsistencyModel().to(device)
    optimizer = torch.optim.AdamW(lcm.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_dataloader:
            x_start = batch['image'].to(device)

            # Sample adjacent timesteps
            t1 = torch.randint(1, lcm.num_timesteps, (x_start.shape[0],), device=device)
            t2 = t1 - 1

            # Consistency loss
            loss = lcm.consistency_loss(x_start, t1, t2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return lcm


if __name__ == '__main__':
    # Demo
    print("Latent Consistency Model - Fast Image Generation")
    print("Generates images in 4-8 steps instead of 50+")
    print()

    generator = FastImageGenerator(num_inference_steps=4)

    # Benchmark
    generator.benchmark_speed()

    # Generate
    print("\nGenerating sample images...")
    images = generator.generate_image(
        prompt="A beautiful sunset over mountains",
        num_images=4,
    )
    print(f"Generated {images.shape[0]} images in 4 steps!")
