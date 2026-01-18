import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class VisionEncoder(nn.Module):
    """encodes images to understand visual content"""

    def __init__(self, embed_dim=768):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

        self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True)
            for _ in range(12)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed[:, :x.size(1), :]

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        return x


class AudioEncoder(nn.Module):
    """encodes audio to understand content"""

    def __init__(self, embed_dim=768):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=10, stride=5)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=4, stride=2)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6
        )

        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.transpose(1, 2)

        x = self.transformer(x)

        x = self.proj(x)

        return x


class MultiModalFusion(nn.Module):
    """fuses different modalities together"""

    def __init__(self, embed_dim=768):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=12, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key_value):
        attn_out, _ = self.cross_attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query


class MultiModalUnderstanding(nn.Module):
    """understands and answers questions about any content"""

    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()

        self.text_encoder = nn.Embedding(vocab_size, embed_dim)
        self.vision_encoder = VisionEncoder(embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim)

        self.vision_fusion = MultiModalFusion(embed_dim)
        self.audio_fusion = MultiModalFusion(embed_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True),
            num_layers=8
        )

        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, text_tokens, image=None, audio=None):
        text_embed = self.text_encoder(text_tokens)

        if image is not None:
            vision_embed = self.vision_encoder(image)
            text_embed = self.vision_fusion(text_embed, vision_embed)

        if audio is not None:
            audio_embed = self.audio_encoder(audio)
            text_embed = self.audio_fusion(text_embed, audio_embed)

        output = self.transformer(text_embed)

        logits = self.output_proj(output)

        return logits

    def generate_answer(self, text_tokens, image=None, audio=None, max_length=200):
        self.eval()

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(text_tokens, image, audio)

                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                text_tokens = torch.cat([text_tokens, next_token], dim=1)

                if next_token.item() == 3:
                    break

        return text_tokens


class ImageCaptioner:
    """generates captions for images"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def caption_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        prompt = "describe this image:"
        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            output_tokens = self.model.generate_answer(text_tensor, image=image_tensor)

        caption = self.tokenizer.decode(output_tokens[0].cpu().tolist())

        return caption


class VisualQuestionAnswering:
    """answers questions about images"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer_question(self, image_path, question):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        full_prompt = f"Question: {question} Answer:"
        text_tokens = self.tokenizer.encode(full_prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            output_tokens = self.model.generate_answer(text_tensor, image=image_tensor)

        answer = self.tokenizer.decode(output_tokens[0].cpu().tolist())

        return answer


class MultiModalQA:
    """answers any type of question using multiple modalities"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.image_captioner = ImageCaptioner(model, tokenizer, device)
        self.vqa = VisualQuestionAnswering(model, tokenizer, device)

    def answer(self, question, image_path=None, audio_path=None, video_path=None):
        """answers questions about any content"""

        image_tensor = None
        audio_tensor = None

        if image_path:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

        if audio_path:
            import wave
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).to(self.device)

        if video_path:
            import cv2
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                image_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
            cap.release()

        text_tokens = self.tokenizer.encode(question)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            output_tokens = self.model.generate_answer(
                text_tensor,
                image=image_tensor,
                audio=audio_tensor
            )

        answer = self.tokenizer.decode(output_tokens[0].cpu().tolist())

        return answer

    def understand_content(self, content_path, content_type='auto'):
        """understands and describes any type of content"""

        if content_type == 'auto':
            if content_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                content_type = 'image'
            elif content_path.lower().endswith(('.mp4', '.avi', '.mov')):
                content_type = 'video'
            elif content_path.lower().endswith(('.wav', '.mp3', '.flac')):
                content_type = 'audio'

        if content_type == 'image':
            return self.image_captioner.caption_image(content_path)

        elif content_type == 'video':
            import cv2
            cap = cv2.VideoCapture(content_path)

            descriptions = []
            frame_count = 0
            sample_rate = 30

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)

                    temp_path = f'/tmp/frame_{frame_count}.jpg'
                    frame_pil.save(temp_path)

                    desc = self.image_captioner.caption_image(temp_path)
                    descriptions.append(desc)

                frame_count += 1

            cap.release()

            return f"Video contains {frame_count} frames. Sample descriptions: " + " | ".join(descriptions[:5])

        elif content_type == 'audio':
            return "Audio content detected. Analyzing audio features..."

        return "Unknown content type"
