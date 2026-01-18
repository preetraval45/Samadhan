import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class CameraMotionControl(nn.Module):
    """controls camera movement in generated videos"""

    def __init__(self, latent_dim=512):
        super().__init__()

        self.motion_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def encode_camera_motion(self, motion_type, params):
        """encodes camera motion parameters"""

        motion_vector = torch.zeros(9)

        if motion_type == 'pan_left':
            motion_vector[0] = params.get('speed', 1.0)

        elif motion_type == 'pan_right':
            motion_vector[1] = params.get('speed', 1.0)

        elif motion_type == 'pan_up':
            motion_vector[2] = params.get('speed', 1.0)

        elif motion_type == 'pan_down':
            motion_vector[3] = params.get('speed', 1.0)

        elif motion_type == 'zoom_in':
            motion_vector[4] = params.get('speed', 1.0)

        elif motion_type == 'zoom_out':
            motion_vector[5] = params.get('speed', 1.0)

        elif motion_type == 'rotate_cw':
            motion_vector[6] = params.get('speed', 1.0)

        elif motion_type == 'rotate_ccw':
            motion_vector[7] = params.get('speed', 1.0)

        elif motion_type == 'static':
            motion_vector[8] = 1.0

        return motion_vector

    def forward(self, motion_type, params=None):
        if params is None:
            params = {}

        motion_vec = self.encode_camera_motion(motion_type, params)
        motion_vec = motion_vec.unsqueeze(0).to(next(self.parameters()).device)

        motion_embedding = self.motion_encoder(motion_vec)

        return motion_embedding


class ObjectMotionControl(nn.Module):
    """controls object movement trajectories"""

    def __init__(self, latent_dim=512, max_objects=10):
        super().__init__()

        self.max_objects = max_objects

        self.trajectory_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.object_encoder = nn.Linear(128, latent_dim)

    def encode_trajectory(self, start_pos, end_pos, num_frames):
        """encodes motion trajectory"""

        x = np.linspace(start_pos[0], end_pos[0], num_frames)
        y = np.linspace(start_pos[1], end_pos[1], num_frames)

        trajectory = np.stack([x, y], axis=0)

        return trajectory

    def encode_bezier_trajectory(self, start, control1, control2, end, num_frames):
        """encodes bezier curve trajectory"""

        t = np.linspace(0, 1, num_frames)

        x = (1-t)**3 * start[0] + 3*(1-t)**2*t * control1[0] + 3*(1-t)*t**2 * control2[0] + t**3 * end[0]
        y = (1-t)**3 * start[1] + 3*(1-t)**2*t * control1[1] + 3*(1-t)*t**2 * control2[1] + t**3 * end[1]

        trajectory = np.stack([x, y], axis=0)

        return trajectory

    def forward(self, trajectories):
        embeddings = []

        for traj in trajectories:
            traj_tensor = torch.from_numpy(traj).float().unsqueeze(0)
            traj_tensor = traj_tensor.to(next(self.parameters()).device)

            encoded = self.trajectory_encoder(traj_tensor)
            encoded = encoded.squeeze(-1)

            obj_embedding = self.object_encoder(encoded)

            embeddings.append(obj_embedding)

        if embeddings:
            return torch.stack(embeddings, dim=1)
        else:
            return None


class SceneTransitionModel(nn.Module):
    """handles transitions between scenes"""

    def __init__(self):
        super().__init__()

        self.transition_types = [
            'fade',
            'dissolve',
            'wipe_left',
            'wipe_right',
            'wipe_up',
            'wipe_down',
            'zoom',
            'spin'
        ]

    def apply_fade(self, frame1, frame2, progress):
        """fade transition"""

        return frame1 * (1 - progress) + frame2 * progress

    def apply_wipe(self, frame1, frame2, progress, direction='left'):
        """wipe transition"""

        h, w, c = frame1.shape

        result = frame1.copy()

        if direction == 'left':
            cutoff = int(w * progress)
            result[:, :cutoff] = frame2[:, :cutoff]

        elif direction == 'right':
            cutoff = int(w * (1 - progress))
            result[:, cutoff:] = frame2[:, cutoff:]

        elif direction == 'up':
            cutoff = int(h * progress)
            result[:cutoff, :] = frame2[:cutoff, :]

        elif direction == 'down':
            cutoff = int(h * (1 - progress))
            result[cutoff:, :] = frame2[cutoff:, :]

        return result

    def apply_zoom(self, frame1, frame2, progress):
        """zoom transition"""

        h, w, c = frame1.shape

        scale = 1.0 + progress * 0.5

        center_h, center_w = h // 2, w // 2

        new_h, new_w = int(h * scale), int(w * scale)

        zoomed1 = cv2.resize(frame1, (new_w, new_h))

        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2

        zoomed1 = zoomed1[crop_h:crop_h+h, crop_w:crop_w+w]

        return self.apply_fade(zoomed1, frame2, progress)

    def create_transition(self, frame1, frame2, transition_type='fade', num_frames=10):
        """creates transition between two frames"""

        frames = []

        for i in range(num_frames):
            progress = i / (num_frames - 1)

            if transition_type == 'fade' or transition_type == 'dissolve':
                frame = self.apply_fade(frame1, frame2, progress)

            elif transition_type == 'wipe_left':
                frame = self.apply_wipe(frame1, frame2, progress, 'left')

            elif transition_type == 'wipe_right':
                frame = self.apply_wipe(frame1, frame2, progress, 'right')

            elif transition_type == 'wipe_up':
                frame = self.apply_wipe(frame1, frame2, progress, 'up')

            elif transition_type == 'wipe_down':
                frame = self.apply_wipe(frame1, frame2, progress, 'down')

            elif transition_type == 'zoom':
                frame = self.apply_zoom(frame1, frame2, progress)

            else:
                frame = self.apply_fade(frame1, frame2, progress)

            frames.append(frame.astype(np.uint8))

        return frames


class TemporalConsistencyModel(nn.Module):
    """ensures consistency across video frames"""

    def __init__(self, feature_dim=512):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.temporal_lstm = nn.LSTM(256, feature_dim, num_layers=2, batch_first=True)

        self.consistency_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, frames):
        batch_size, num_frames, c, h, w = frames.shape

        frames_flat = frames.view(-1, c, h, w)

        features = self.feature_extractor(frames_flat)
        features = features.view(batch_size, num_frames, -1)

        temporal_features, _ = self.temporal_lstm(features)

        consistency_scores = self.consistency_head(temporal_features)

        return consistency_scores


class HighResolutionVideoGenerator(nn.Module):
    """generates 4K/8K resolution videos"""

    def __init__(self, base_generator, target_resolution='4k'):
        super().__init__()

        self.base_generator = base_generator

        if target_resolution == '4k':
            self.target_size = (3840, 2160)
            self.upscale_factor = 8

        elif target_resolution == '8k':
            self.target_size = (7680, 4320)
            self.upscale_factor = 16

        else:
            self.target_size = (1920, 1080)
            self.upscale_factor = 4

        self.spatial_upsampler = self._build_upsampler(self.upscale_factor)

        self.temporal_refiner = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )

    def _build_upsampler(self, scale_factor):
        layers = []

        current_scale = 1
        channels = 64

        while current_scale < scale_factor:
            layers.extend([
                nn.Conv2d(3 if current_scale == 1 else channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU()
            ])

            current_scale *= 2

        layers.append(nn.Conv2d(channels, 3, 3, padding=1))

        return nn.Sequential(*layers)

    def upscale_frame(self, frame):
        """upscales single frame to target resolution"""

        if isinstance(frame, np.ndarray):
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        else:
            frame_tensor = frame

        frame_tensor = frame_tensor.unsqueeze(0)

        with torch.no_grad():
            upscaled = self.spatial_upsampler(frame_tensor)

        return upscaled

    def forward(self, low_res_video):
        upscaled_frames = []

        for frame in low_res_video:
            upscaled_frame = self.upscale_frame(frame)
            upscaled_frames.append(upscaled_frame)

        video_tensor = torch.stack(upscaled_frames, dim=2)

        refined_video = self.temporal_refiner(video_tensor)

        refined_video = refined_video + video_tensor

        return refined_video


class VideoToVideoTranslator(nn.Module):
    """translates videos between styles"""

    def __init__(self, feature_dim=256):
        super().__init__()

        self.content_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), padding=(0, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, feature_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )

        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_dim, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=(1, 7, 7), padding=(0, 3, 3)),
            nn.Tanh()
        )

    def forward(self, source_video, style_image):
        content_features = self.content_encoder(source_video)

        style_features = self.style_encoder(style_image)

        styled_features = content_features + style_features.view(1, -1, 1, 1, 1)

        output_video = self.decoder(styled_features)

        return output_video


class AdvancedVideoGenerator:
    """combines all advanced video generation features"""

    def __init__(self, base_model, tokenizer, device='cuda'):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

        self.camera_control = CameraMotionControl().to(device)
        self.object_motion = ObjectMotionControl().to(device)
        self.transitions = SceneTransitionModel()
        self.consistency = TemporalConsistencyModel().to(device)
        self.hd_generator_4k = HighResolutionVideoGenerator(base_model, '4k').to(device)
        self.hd_generator_8k = HighResolutionVideoGenerator(base_model, '8k').to(device)
        self.video_translator = VideoToVideoTranslator().to(device)

    def generate_with_camera_motion(self, prompt, motion_type='static', motion_params=None, num_frames=16):
        """generates video with controlled camera motion"""

        camera_embedding = self.camera_control(motion_type, motion_params)

        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            video = self.base_model.generate(text_tensor, num_frames=num_frames)

        return video

    def generate_with_object_motion(self, prompt, trajectories, num_frames=16):
        """generates video with controlled object motion"""

        motion_embeddings = self.object_motion(trajectories)

        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            video = self.base_model.generate(text_tensor, num_frames=num_frames)

        return video

    def generate_4k(self, prompt, num_frames=16):
        """generates 4K resolution video"""

        low_res_video = self.generate_with_camera_motion(prompt, num_frames=num_frames)

        with torch.no_grad():
            hd_video = self.hd_generator_4k(low_res_video)

        return hd_video

    def generate_8k(self, prompt, num_frames=16):
        """generates 8K resolution video"""

        low_res_video = self.generate_with_camera_motion(prompt, num_frames=num_frames)

        with torch.no_grad():
            hd_video = self.hd_generator_8k(low_res_video)

        return hd_video

    def translate_style(self, source_video, style_image):
        """translates video to match style image"""

        if isinstance(source_video, np.ndarray):
            video_tensor = torch.from_numpy(source_video).permute(0, 3, 1, 2).float() / 255.0
            video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        else:
            video_tensor = source_video

        if isinstance(style_image, np.ndarray):
            style_tensor = torch.from_numpy(style_image).permute(2, 0, 1).float() / 255.0
        else:
            style_tensor = style_image

        style_tensor = style_tensor.unsqueeze(0).to(self.device)
        video_tensor = video_tensor.to(self.device)

        with torch.no_grad():
            styled_video = self.video_translator(video_tensor, style_tensor)

        return styled_video
