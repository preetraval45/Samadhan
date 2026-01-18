import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, t, h, w = x.shape

        x = x.permute(0, 2, 3, 4, 1).reshape(b * t * h * w, c)

        qkv = self.qkv(x).reshape(b * t * h * w, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b * t * h * w, c)
        out = self.proj(out)

        out = out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return out


class VideoResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv3DBlock(channels, channels)
        self.conv2 = Conv3DBlock(channels, channels)
        self.temporal_attn = TemporalAttention(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.temporal_attn(x)
        return x + residual


class VideoEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()

        self.conv_in = Conv3DBlock(in_channels, 128)

        self.down1 = nn.Sequential(
            VideoResBlock(128),
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.down2 = nn.Sequential(
            VideoResBlock(256),
            nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.mid = nn.Sequential(
            VideoResBlock(512),
            VideoResBlock(512)
        )

        self.conv_out = nn.Conv3d(512, latent_dim, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.mid(x)
        x = self.conv_out(x)
        return x


class VideoDecoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()

        self.conv_in = Conv3DBlock(latent_dim, 512)

        self.mid = nn.Sequential(
            VideoResBlock(512),
            VideoResBlock(512)
        )

        self.up1 = nn.Sequential(
            VideoResBlock(512),
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(512, 256, kernel_size=3, padding=1)
        )

        self.up2 = nn.Sequential(
            VideoResBlock(256),
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(256, 128, kernel_size=3, padding=1)
        )

        self.conv_out = nn.Conv3d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.conv_out(x)
        x = torch.tanh(x)
        return x


class VideoTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=6
        )

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_tokens):
        x = self.embedding(text_tokens)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.proj(x)

        return x


class VideoDiffusionModel(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.text_proj = nn.Linear(512, latent_dim)

        self.conv_in = Conv3DBlock(latent_dim, latent_dim)

        self.blocks = nn.ModuleList([
            VideoResBlock(latent_dim),
            VideoResBlock(latent_dim),
            VideoResBlock(latent_dim),
            VideoResBlock(latent_dim)
        ])

        self.conv_out = nn.Conv3d(latent_dim, latent_dim, 1)

    def forward(self, x, timestep, text_embed):
        t_emb = self.get_timestep_embedding(timestep, x.device)
        t_emb = self.time_embed(t_emb)
        t_emb = t_emb.view(-1, x.size(1), 1, 1, 1)

        text_emb = self.text_proj(text_embed)
        text_emb = text_emb.view(-1, x.size(1), 1, 1, 1)

        x = x + t_emb + text_emb

        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x

    def get_timestep_embedding(self, timesteps, device):
        half_dim = 128
        emb = torch.exp(torch.arange(half_dim, device=device) * -(np.log(10000) / half_dim))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CustomVideoGenerator(nn.Module):
    def __init__(self, vocab_size, latent_dim=512):
        super().__init__()

        self.text_encoder = VideoTextEncoder(vocab_size)
        self.diffusion = VideoDiffusionModel(latent_dim)
        self.decoder = VideoDecoder(latent_dim)

        self.latent_dim = latent_dim
        self.num_timesteps = 1000

    @torch.no_grad()
    def generate(self, text_tokens, num_frames=16, height=256, width=256, num_steps=50):
        self.eval()
        device = next(self.parameters()).device

        batch_size = text_tokens.size(0)

        text_embed = self.text_encoder(text_tokens)

        latent_h, latent_w = height // 4, width // 4
        latents = torch.randn(batch_size, self.latent_dim, num_frames, latent_h, latent_w, device=device)

        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()

        for t in timesteps:
            t_batch = t.unsqueeze(0).repeat(batch_size)

            noise_pred = self.diffusion(latents, t_batch, text_embed)

            beta = 0.0001 + (0.02 - 0.0001) * t / self.num_timesteps
            alpha = 1 - beta

            if t > 0:
                noise = torch.randn_like(latents)
            else:
                noise = torch.zeros_like(latents)

            latents = (latents - beta * noise_pred) / torch.sqrt(alpha)
            latents = latents + torch.sqrt(beta) * noise

        videos = self.decoder(latents)

        videos = (videos + 1) / 2
        videos = videos.clamp(0, 1)

        return videos


class VideoGeneratorInference:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate_video(self, prompt, num_frames=16, height=256, width=256, fps=8):
        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            videos = self.model.generate(text_tensor, num_frames, height, width)

        video = videos[0].cpu().permute(1, 2, 3, 0).numpy()
        video = (video * 255).astype(np.uint8)

        return video

    def save_video(self, video, filepath, fps=8):
        height, width = video.shape[1:3]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        for frame in video:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def save_gif(self, video, filepath, fps=8):
        frames = [Image.fromarray(frame) for frame in video]
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,
            loop=0
        )

    def generate_long_video(self, prompt, total_frames=120, chunk_size=16, overlap=4):
        """generates long videos by stitching chunks together"""
        all_frames = []

        num_chunks = (total_frames - overlap) // (chunk_size - overlap) + 1

        for i in range(num_chunks):
            chunk_prompt = f"{prompt} (segment {i+1})"

            video_chunk = self.generate_video(prompt, num_frames=chunk_size)

            if i == 0:
                all_frames.extend(video_chunk)
            else:
                all_frames.extend(video_chunk[overlap:])

            if len(all_frames) >= total_frames:
                break

        return np.array(all_frames[:total_frames])


class VideoInterpolator:
    """interpolates between frames to increase fps"""

    def __init__(self):
        pass

    def interpolate(self, video, target_fps_multiplier=2):
        interpolated = []

        for i in range(len(video) - 1):
            interpolated.append(video[i])

            for j in range(1, target_fps_multiplier):
                alpha = j / target_fps_multiplier
                interpolated_frame = (1 - alpha) * video[i] + alpha * video[i + 1]
                interpolated.append(interpolated_frame.astype(np.uint8))

        interpolated.append(video[-1])

        return np.array(interpolated)
