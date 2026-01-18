import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class FaceEncoder(nn.Module):
    """extracts and preserves facial features"""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class FacePreservingDecoder(nn.Module):
    """reconstructs images while preserving facial features"""

    def __init__(self, latent_dim=512):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 16, 16)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return x


class FaceDetector:
    """detects faces in images"""

    def __init__(self):
        pass

    def detect_face(self, image):
        """detects face region in image using simple heuristics"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape

        center_y, center_x = h // 2, w // 2

        face_h = int(h * 0.6)
        face_w = int(w * 0.5)

        y1 = max(0, center_y - face_h // 2)
        y2 = min(h, center_y + face_h // 2)
        x1 = max(0, center_x - face_w // 2)
        x2 = min(w, center_x + face_w // 2)

        edges = cv2.Canny(gray, 100, 200)
        face_region_edges = edges[y1:y2, x1:x2]

        if np.sum(face_region_edges) > 1000:
            return (x1, y1, x2, y2)

        return None

    def extract_face(self, image, bbox):
        """extracts face region from image"""
        x1, y1, x2, y2 = bbox

        if isinstance(image, Image.Image):
            image = np.array(image)

        face = image[y1:y2, x1:x2]

        return face


class FacePreservingGenerator(nn.Module):
    """generates images while preserving uploaded face"""

    def __init__(self, vocab_size, latent_dim=512):
        super().__init__()

        self.face_encoder = FaceEncoder()
        self.decoder = FacePreservingDecoder(latent_dim)

        self.text_encoder = nn.Sequential(
            nn.Embedding(vocab_size, 512),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
                num_layers=4
            )
        )

        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, face_image, text_tokens):
        face_features = self.face_encoder(face_image)

        text_embed = self.text_encoder(text_tokens)
        text_features = text_embed.mean(dim=1)

        combined = torch.cat([face_features, text_features], dim=1)
        latent = self.fusion(combined)

        output = self.decoder(latent)

        return output


class FaceBlender:
    """blends original face into generated content"""

    def __init__(self):
        self.detector = FaceDetector()

    def blend_face(self, original_image, generated_image, blend_strength=0.8):
        """blends original face into generated image"""

        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        if isinstance(generated_image, Image.Image):
            generated_image = np.array(generated_image)

        face_bbox = self.detector.detect_face(original_image)

        if face_bbox is None:
            return generated_image

        x1, y1, x2, y2 = face_bbox

        original_face = original_image[y1:y2, x1:x2]

        gen_h, gen_w = generated_image.shape[:2]

        if y2 <= gen_h and x2 <= gen_w:
            gen_face_region = generated_image[y1:y2, x1:x2]

            if gen_face_region.shape == original_face.shape:
                mask = self.create_smooth_mask(original_face.shape[:2])

                blended_face = (
                    original_face * mask * blend_strength +
                    gen_face_region * (1 - mask * blend_strength)
                ).astype(np.uint8)

                result = generated_image.copy()
                result[y1:y2, x1:x2] = blended_face

                return result

        return generated_image

    def create_smooth_mask(self, shape):
        """creates smooth blending mask"""
        h, w = shape
        mask = np.ones((h, w, 1))

        fade_size = min(h, w) // 8

        for i in range(fade_size):
            alpha = i / fade_size

            mask[i, :] = alpha
            mask[h-1-i, :] = alpha
            mask[:, i] = alpha
            mask[:, w-1-i] = alpha

        return mask


class VideoFacePreserver:
    """preserves face across video frames"""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_blender = FaceBlender()

    def preserve_face_in_video(self, reference_face, generated_video, blend_strength=0.9):
        """applies face preservation to all video frames"""

        if isinstance(reference_face, Image.Image):
            reference_face = np.array(reference_face)

        preserved_frames = []

        for frame in generated_video:
            blended_frame = self.face_blender.blend_face(
                reference_face,
                frame,
                blend_strength=blend_strength
            )
            preserved_frames.append(blended_frame)

        return np.array(preserved_frames)

    def track_and_preserve_face(self, reference_face, generated_video):
        """tracks face movement and preserves it across frames"""

        face_bbox = self.face_detector.detect_face(reference_face)

        if face_bbox is None:
            return generated_video

        preserved_frames = []

        for frame in generated_video:
            frame_bbox = self.face_detector.detect_face(frame)

            if frame_bbox is not None:
                blended = self.face_blender.blend_face(reference_face, frame)
                preserved_frames.append(blended)
            else:
                preserved_frames.append(frame)

        return np.array(preserved_frames)


class FacePreservingInference:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.face_blender = FaceBlender()
        self.video_preserver = VideoFacePreserver()

    def generate_with_face(self, prompt, face_image_path, preserve_face=True):
        """generates image with face preservation"""

        face_image = Image.open(face_image_path).convert('RGB')
        face_image = face_image.resize((256, 256))

        face_tensor = torch.from_numpy(np.array(face_image)).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        face_tensor = (face_tensor - 0.5) / 0.5

        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            generated = self.model(face_tensor, text_tensor)

        generated_image = generated[0].cpu().permute(1, 2, 0).numpy()
        generated_image = ((generated_image * 0.5 + 0.5) * 255).astype(np.uint8)

        if preserve_face:
            original_face = np.array(face_image)
            generated_image = self.face_blender.blend_face(
                original_face,
                generated_image,
                blend_strength=0.85
            )

        return Image.fromarray(generated_image)

    def generate_video_with_face(self, prompt, face_image_path, num_frames=16, preserve_face=True):
        """generates video while preserving uploaded face"""

        face_image = Image.open(face_image_path).convert('RGB')
        face_array = np.array(face_image.resize((256, 256)))

        from .video_generator import VideoGeneratorInference

        video_generator = VideoGeneratorInference(
            self.model,
            self.tokenizer,
            self.device
        )

        generated_video = video_generator.generate_video(prompt, num_frames)

        if preserve_face:
            generated_video = self.video_preserver.preserve_face_in_video(
                face_array,
                generated_video,
                blend_strength=0.9
            )

        return generated_video
