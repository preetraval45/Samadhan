import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class FacialLandmarkDetector(nn.Module):
    """detects 68 facial landmarks"""

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.landmark_head = nn.Linear(512, 68 * 2)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        landmarks = self.landmark_head(features)
        landmarks = landmarks.view(-1, 68, 2)

        return landmarks


class ExpressionEncoder(nn.Module):
    """encodes facial expressions"""

    def __init__(self, expression_dim=128):
        super().__init__()

        self.landmark_encoder = nn.Sequential(
            nn.Linear(68 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, expression_dim)
        )

        self.expression_types = [
            'neutral', 'happy', 'sad', 'angry', 'surprised',
            'disgusted', 'fearful', 'contemptuous'
        ]

    def forward(self, landmarks):
        landmarks_flat = landmarks.view(landmarks.size(0), -1)

        expression_code = self.landmark_encoder(landmarks_flat)

        return expression_code


class ExpressionTransferModel(nn.Module):
    """transfers expressions between faces"""

    def __init__(self, expression_dim=128):
        super().__init__()

        self.source_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.expression_encoder = ExpressionEncoder(expression_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + expression_dim, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )

        self.landmark_detector = FacialLandmarkDetector()

    def forward(self, source_face, target_expression):
        source_features = self.source_encoder(source_face)

        target_landmarks = self.landmark_detector(target_expression)
        expression_code = self.expression_encoder(target_landmarks)

        expression_broadcast = expression_code.view(
            expression_code.size(0), -1, 1, 1
        ).repeat(1, 1, source_features.size(2), source_features.size(3))

        combined = torch.cat([source_features, expression_broadcast], dim=1)

        output = self.decoder(combined)

        return output


class AgeProgressionModel(nn.Module):
    """age progression and regression"""

    def __init__(self, num_age_groups=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.age_embedding = nn.Embedding(num_age_groups, 512)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, face, target_age_group):
        features = self.encoder(face)

        age_emb = self.age_embedding(target_age_group)
        age_emb = age_emb.view(age_emb.size(0), -1, 1, 1)
        age_emb = age_emb.repeat(1, 1, features.size(2), features.size(3))

        combined = torch.cat([features, age_emb], dim=1)

        output = self.decoder(combined)

        return output


class GenderSwapModel(nn.Module):
    """swaps gender in faces"""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.gender_transform = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, face):
        features = self.encoder(face)

        transformed = self.gender_transform(features)

        output = self.decoder(transformed)

        return output


class RealTimeDeepfake(nn.Module):
    """real-time deepfake at 30fps+"""

    def __init__(self):
        super().__init__()

        self.lightweight_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.face_swap_core = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.lightweight_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, source, target):
        source_features = self.lightweight_encoder(source)
        target_features = self.lightweight_encoder(target)

        swapped = self.face_swap_core(source_features + target_features)

        output = self.lightweight_decoder(swapped)

        return output


class LipSyncModel(nn.Module):
    """lip sync for any language"""

    def __init__(self):
        super().__init__()

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(80, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU()
        )

        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.lip_generator = nn.Sequential(
            nn.ConvTranspose2d(128 + 512, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, face, audio_features):
        face_features = self.face_encoder(face)

        audio_encoded = self.audio_encoder(audio_features)

        audio_2d = audio_encoded.unsqueeze(-1).repeat(
            1, 1, face_features.size(2), face_features.size(3)
        )

        combined = torch.cat([face_features, audio_2d], dim=1)

        output = self.lip_generator(combined)

        return output


class FullBodyDeepfake(nn.Module):
    """full body deepfake not just face"""

    def __init__(self):
        super().__init__()

        self.body_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.pose_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.body_decoder = nn.Sequential(
            nn.ConvTranspose2d(640, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, source_body, target_pose):
        body_features = self.body_encoder(source_body)

        pose_features = self.pose_encoder(target_pose)

        pose_upsampled = F.interpolate(
            pose_features,
            size=(body_features.size(2), body_features.size(3)),
            mode='bilinear'
        )

        combined = torch.cat([body_features, pose_upsampled], dim=1)

        output = self.body_decoder(combined)

        return output


class DeepfakeWatermarking:
    """adds watermarks to deepfakes for ethical use"""

    @staticmethod
    def add_invisible_watermark(image, watermark_data="DEEPFAKE"):
        """adds invisible watermark"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        watermark_binary = ''.join(format(ord(c), '08b') for c in watermark_data)

        h, w, c = image.shape

        watermarked = image.copy().astype(np.int16)

        bit_idx = 0

        for i in range(min(h, 100)):
            for j in range(min(w, 100)):
                if bit_idx < len(watermark_binary):
                    bit = int(watermark_binary[bit_idx])

                    watermarked[i, j, 0] = (watermarked[i, j, 0] & ~1) | bit

                    bit_idx += 1
                else:
                    break

        return watermarked.astype(np.uint8)

    @staticmethod
    def add_visible_watermark(image, text="SYNTHETIC", position='bottom_right'):
        """adds visible watermark"""

        if isinstance(image, Image.Image):
            image_pil = image
        else:
            image_pil = Image.fromarray(image)

        watermarked = np.array(image_pil).copy()

        h, w = watermarked.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        if position == 'bottom_right':
            x = w - text_size[0] - 20
            y = h - 20
        elif position == 'bottom_left':
            x = 20
            y = h - 20
        elif position == 'top_right':
            x = w - text_size[0] - 20
            y = 30
        else:
            x = 20
            y = 30

        cv2.rectangle(
            watermarked,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            watermarked,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        return watermarked


class AdvancedDeepfakeSystem:
    """combines all deepfake capabilities"""

    def __init__(self, device='cuda'):
        self.device = device

        self.expression_transfer = ExpressionTransferModel().to(device)
        self.age_model = AgeProgressionModel().to(device)
        self.gender_swap = GenderSwapModel().to(device)
        self.realtime = RealTimeDeepfake().to(device)
        self.lip_sync = LipSyncModel().to(device)
        self.fullbody = FullBodyDeepfake().to(device)

        self.watermarking = DeepfakeWatermarking()

    def transfer_expression(self, source_face, target_expression):
        """transfers expression from target to source"""

        source_tensor = self._prepare_image(source_face)
        target_tensor = self._prepare_image(target_expression)

        with torch.no_grad():
            result = self.expression_transfer(source_tensor, target_tensor)

        return self._tensor_to_image(result)

    def progress_age(self, face, target_age_group):
        """ages face to target age group"""

        face_tensor = self._prepare_image(face)
        age_tensor = torch.tensor([target_age_group], device=self.device)

        with torch.no_grad():
            result = self.age_model(face_tensor, age_tensor)

        return self._tensor_to_image(result)

    def swap_gender(self, face):
        """swaps gender in face"""

        face_tensor = self._prepare_image(face)

        with torch.no_grad():
            result = self.gender_swap(face_tensor)

        return self._tensor_to_image(result)

    def realtime_swap(self, source, target):
        """performs real-time face swap"""

        source_tensor = self._prepare_image(source)
        target_tensor = self._prepare_image(target)

        with torch.no_grad():
            result = self.realtime(source_tensor, target_tensor)

        return self._tensor_to_image(result)

    def sync_lips(self, face, audio_features):
        """syncs lips to audio"""

        face_tensor = self._prepare_image(face)

        if isinstance(audio_features, np.ndarray):
            audio_tensor = torch.from_numpy(audio_features).float().to(self.device)
        else:
            audio_tensor = audio_features

        with torch.no_grad():
            result = self.lip_sync(face_tensor, audio_tensor)

        return self._tensor_to_image(result)

    def fullbody_deepfake(self, source_body, target_pose):
        """creates full body deepfake"""

        source_tensor = self._prepare_image(source_body)
        pose_tensor = self._prepare_image(target_pose)

        with torch.no_grad():
            result = self.fullbody(source_tensor, pose_tensor)

        return self._tensor_to_image(result)

    def _prepare_image(self, image):
        """prepares image for model input"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor

    def _tensor_to_image(self, tensor):
        """converts tensor to image"""

        image = tensor[0].cpu().permute(1, 2, 0).numpy()
        image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)

        return Image.fromarray(image)
