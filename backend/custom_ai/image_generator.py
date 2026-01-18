import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.relu(x + residual)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w).permute(0, 2, 1)

        attn = torch.bmm(q, k)
        attn = attn * (c ** -0.5)
        attn = F.softmax(attn, dim=2)

        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)

        return x + out


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)

        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(128),
                ResidualBlock(128),
                nn.Conv2d(128, 256, 3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(256),
                ResidualBlock(256),
                nn.Conv2d(256, 512, 3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(512),
                ResidualBlock(512),
                nn.Conv2d(512, 512, 3, stride=2, padding=1)
            )
        ])

        self.mid = nn.Sequential(
            ResidualBlock(512),
            AttentionBlock(512),
            ResidualBlock(512)
        )

        self.norm_out = nn.GroupNorm(8, 512)
        self.conv_out = nn.Conv2d(512, latent_dim, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        for down in self.down_blocks:
            x = down(x)

        x = self.mid(x)
        x = self.norm_out(x)
        x = F.relu(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()

        self.conv_in = nn.Conv2d(latent_dim, 512, 3, padding=1)

        self.mid = nn.Sequential(
            ResidualBlock(512),
            AttentionBlock(512),
            ResidualBlock(512)
        )

        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(512),
                ResidualBlock(512),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 512, 3, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(512),
                ResidualBlock(512),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, 3, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(256),
                ResidualBlock(256),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 128, 3, padding=1)
            )
        ])

        self.norm_out = nn.GroupNorm(8, 128)
        self.conv_out = nn.Conv2d(128, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)

        for up in self.up_blocks:
            x = up(x)

        x = self.norm_out(x)
        x = F.relu(x)
        x = self.conv_out(x)
        x = torch.tanh(x)

        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, output_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=4
        )

        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, text_tokens):
        x = self.embedding(text_tokens)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.proj(x)

        return x


class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )

        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
                ResidualBlock(latent_dim),
                ResidualBlock(latent_dim)
            ),
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim * 2, 3, stride=2, padding=1),
                ResidualBlock(latent_dim * 2),
                ResidualBlock(latent_dim * 2)
            )
        ])

        self.mid = nn.Sequential(
            ResidualBlock(latent_dim * 2),
            AttentionBlock(latent_dim * 2),
            ResidualBlock(latent_dim * 2)
        )

        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(latent_dim * 2),
                ResidualBlock(latent_dim * 2),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(latent_dim * 2, latent_dim, 3, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(latent_dim),
                ResidualBlock(latent_dim),
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
            )
        ])

        self.out = nn.Conv2d(latent_dim, latent_dim, 1)

    def forward(self, x, t, text_embed):
        t_embed = self.time_embed(self.get_timestep_embedding(t, x.device))
        t_embed = t_embed.view(-1, x.size(1), 1, 1)

        x = x + t_embed

        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.mid(x)

        for up in self.up_blocks:
            if skips:
                skip = skips.pop()
                x = x + skip
            x = up(x)

        x = self.out(x)
        return x

    def get_timestep_embedding(self, timesteps, device):
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CustomImageGenerator(nn.Module):
    def __init__(self, vocab_size, latent_dim=256):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size, output_dim=latent_dim)
        self.diffusion = DiffusionModel(latent_dim)
        self.decoder = Decoder(latent_dim)

        self.latent_dim = latent_dim
        self.num_timesteps = 1000

    def forward(self, text_tokens, noisy_latents, timestep):
        text_embed = self.text_encoder(text_tokens)
        noise_pred = self.diffusion(noisy_latents, timestep, text_embed)
        return noise_pred

    @torch.no_grad()
    def generate(self, text_tokens, height=256, width=256, num_steps=50, guidance_scale=7.5):
        self.eval()
        device = next(self.parameters()).device

        batch_size = text_tokens.size(0)

        text_embed = self.text_encoder(text_tokens)

        latent_h, latent_w = height // 8, width // 8
        latents = torch.randn(batch_size, self.latent_dim, latent_h, latent_w, device=device)

        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()

        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).repeat(batch_size)

            noise_pred = self.diffusion(latents, t_batch, text_embed)

            beta = self.get_beta_schedule(t, device)
            alpha = 1 - beta
            alpha_bar = self.get_alpha_bar(t, device)

            if t > 0:
                noise = torch.randn_like(latents)
            else:
                noise = torch.zeros_like(latents)

            latents = (latents - beta / torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
            latents = latents + torch.sqrt(beta) * noise

        images = self.decoder(latents)

        images = (images + 1) / 2
        images = images.clamp(0, 1)

        return images

    def get_beta_schedule(self, t, device):
        beta_start = 0.0001
        beta_end = 0.02
        return beta_start + (beta_end - beta_start) * t / self.num_timesteps

    def get_alpha_bar(self, t, device):
        betas = torch.linspace(0.0001, 0.02, self.num_timesteps, device=device)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        return alpha_bar[t]


class ImageGeneratorInference:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate_image(self, prompt, height=512, width=512, num_steps=50):
        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            images = self.model.generate(text_tensor, height, width, num_steps)

        image = images[0].cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)

        return Image.fromarray(image)

    def generate_batch(self, prompts, height=512, width=512):
        images = []
        for prompt in prompts:
            img = self.generate_image(prompt, height, width)
            images.append(img)
        return images

    def save_image(self, image, filepath):
        image.save(filepath)
