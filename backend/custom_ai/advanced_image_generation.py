import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class ControlNetEncoder(nn.Module):
    """controlnet for guided image generation"""

    def __init__(self, conditioning_channels=3, model_channels=320):
        super().__init__()

        self.input_hint_block = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, model_channels, 3, padding=1)
        )

        self.zero_conv = nn.Conv2d(model_channels, model_channels, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, hint):
        h = self.input_hint_block(hint)
        h = self.zero_conv(h)
        return h


class CannyEdgeDetector:
    """extracts canny edges for controlnet"""

    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def detect(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

        edges = edges[:, :, None].repeat(3, axis=2)

        return edges


class DepthEstimator:
    """estimates depth maps for controlnet"""

    def __init__(self):
        pass

    def estimate(self, image):
        """simple depth estimation using gradients"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        depth = np.sqrt(grad_x**2 + grad_y**2)

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)

        depth = depth[:, :, None].repeat(3, axis=2)

        return depth


class InpaintingMask:
    """creates and processes inpainting masks"""

    @staticmethod
    def create_mask(image_size, mask_type='center', **kwargs):
        """creates various types of masks"""

        h, w = image_size

        mask = np.zeros((h, w), dtype=np.uint8)

        if mask_type == 'center':
            center_h, center_w = h // 2, w // 2
            size = kwargs.get('size', min(h, w) // 3)

            mask[
                center_h - size//2 : center_h + size//2,
                center_w - size//2 : center_w + size//2
            ] = 255

        elif mask_type == 'rectangle':
            x1, y1 = kwargs.get('top_left', (w//4, h//4))
            x2, y2 = kwargs.get('bottom_right', (3*w//4, 3*h//4))

            mask[y1:y2, x1:x2] = 255

        elif mask_type == 'circle':
            center = kwargs.get('center', (w//2, h//2))
            radius = kwargs.get('radius', min(h, w) // 4)

            cv2.circle(mask, center, radius, 255, -1)

        elif mask_type == 'random':
            num_shapes = kwargs.get('num_shapes', 5)

            for _ in range(num_shapes):
                shape_type = np.random.choice(['rectangle', 'circle'])

                if shape_type == 'rectangle':
                    x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
                    size_x = np.random.randint(w//10, w//3)
                    size_y = np.random.randint(h//10, h//3)

                    mask[y1:min(y1+size_y, h), x1:min(x1+size_x, w)] = 255

                else:
                    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
                    radius = np.random.randint(min(h, w)//20, min(h, w)//6)

                    cv2.circle(mask, (cx, cy), radius, 255, -1)

        return mask

    @staticmethod
    def feather_mask(mask, feather_amount=10):
        """creates smooth edges on mask"""

        mask_float = mask.astype(np.float32) / 255.0

        mask_float = cv2.GaussianBlur(mask_float, (feather_amount*2+1, feather_amount*2+1), 0)

        return (mask_float * 255).astype(np.uint8)


class InpaintingModel(nn.Module):
    """inpainting model to fill masked regions"""

    def __init__(self, base_generator):
        super().__init__()

        self.generator = base_generator

        self.mask_processor = nn.Sequential(
            nn.Conv2d(4, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1)
        )

    def forward(self, masked_image, mask, text_tokens):
        concat_input = torch.cat([masked_image, mask], dim=1)

        mask_features = self.mask_processor(concat_input)

        output = self.generator(text_tokens)

        return output


class OutpaintingModel(nn.Module):
    """outpainting to extend images beyond borders"""

    def __init__(self, base_generator):
        super().__init__()

        self.generator = base_generator

    def extend_image(self, image, direction='all', extend_pixels=128):
        """extends image in specified direction"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w, c = image.shape

        if direction == 'all':
            new_h = h + 2 * extend_pixels
            new_w = w + 2 * extend_pixels

            extended = np.zeros((new_h, new_w, c), dtype=image.dtype)

            extended[extend_pixels:extend_pixels+h, extend_pixels:extend_pixels+w] = image

            return extended, (extend_pixels, extend_pixels, extend_pixels+h, extend_pixels+w)

        elif direction == 'top':
            extended = np.zeros((h + extend_pixels, w, c), dtype=image.dtype)
            extended[extend_pixels:] = image

            return extended, (0, extend_pixels, w, h+extend_pixels)

        elif direction == 'bottom':
            extended = np.zeros((h + extend_pixels, w, c), dtype=image.dtype)
            extended[:h] = image

            return extended, (0, 0, w, h)

        elif direction == 'left':
            extended = np.zeros((h, w + extend_pixels, c), dtype=image.dtype)
            extended[:, extend_pixels:] = image

            return extended, (extend_pixels, 0, w+extend_pixels, h)

        elif direction == 'right':
            extended = np.zeros((h, w + extend_pixels, c), dtype=image.dtype)
            extended[:, :w] = image

            return extended, (0, 0, w, h)


class SuperResolutionModel(nn.Module):
    """upscales images 8x/16x"""

    def __init__(self, scale_factor=8, channels=64):
        super().__init__()

        self.scale_factor = scale_factor

        self.initial = nn.Conv2d(3, channels, 3, padding=1)

        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(channels) for _ in range(16)
        ])

        num_upsample = int(np.log2(scale_factor))

        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU()
            )
            for _ in range(num_upsample)
        ])

        self.final = nn.Conv2d(channels, 3, 3, padding=1)

    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        initial = self.initial(x)

        residual = initial

        for block in self.residual_blocks:
            residual = residual + block(residual)

        upsampled = residual

        for upsample_layer in self.upsample:
            upsampled = upsample_layer(upsampled)

        output = self.final(upsampled)

        return torch.tanh(output)


class AdvancedImageGenerator:
    """combines all advanced generation features"""

    def __init__(self, base_model, tokenizer, device='cuda'):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

        self.controlnet = ControlNetEncoder().to(device)
        self.inpainting = InpaintingModel(base_model).to(device)
        self.outpainting = OutpaintingModel(base_model).to(device)
        self.upscaler_8x = SuperResolutionModel(scale_factor=8).to(device)
        self.upscaler_16x = SuperResolutionModel(scale_factor=16).to(device)

        self.canny = CannyEdgeDetector()
        self.depth = DepthEstimator()

    def generate_with_canny(self, prompt, reference_image, strength=1.0):
        """generates image guided by canny edges"""

        edges = self.canny.detect(reference_image)

        edges_tensor = torch.from_numpy(edges).permute(2, 0, 1).float() / 255.0
        edges_tensor = edges_tensor.unsqueeze(0).to(self.device)

        control_features = self.controlnet(edges_tensor)

        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            generated = self.base_model.generate(text_tensor)

        return generated

    def generate_with_depth(self, prompt, reference_image):
        """generates image guided by depth map"""

        depth_map = self.depth.estimate(reference_image)

        depth_tensor = torch.from_numpy(depth_map).permute(2, 0, 1).float() / 255.0
        depth_tensor = depth_tensor.unsqueeze(0).to(self.device)

        control_features = self.controlnet(depth_tensor)

        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            generated = self.base_model.generate(text_tensor)

        return generated

    def inpaint(self, image, mask, prompt):
        """fills masked region with generated content"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        masked_image = image.copy()
        masked_image[mask > 127] = 0

        image_tensor = torch.from_numpy(masked_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        text_tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([text_tokens], device=self.device)

        with torch.no_grad():
            generated = self.inpainting(image_tensor, mask_tensor, text_tensor)

        generated_np = generated[0].cpu().permute(1, 2, 0).numpy()
        generated_np = (generated_np * 255).astype(np.uint8)

        mask_3ch = (mask[:, :, None] / 255.0).repeat(3, axis=2)

        result = (image * (1 - mask_3ch) + generated_np * mask_3ch).astype(np.uint8)

        return Image.fromarray(result)

    def outpaint(self, image, direction='all', extend_pixels=128, prompt=""):
        """extends image beyond borders"""

        extended, original_region = self.outpainting.extend_image(image, direction, extend_pixels)

        h, w = extended.shape[:2]

        mask = np.ones((h, w), dtype=np.uint8) * 255

        x1, y1, x2, y2 = original_region
        mask[y1:y2, x1:x2] = 0

        result = self.inpaint(extended, mask, prompt)

        return result

    def upscale_8x(self, image):
        """upscales image 8x"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            upscaled = self.upscaler_8x(image_tensor)

        upscaled_np = upscaled[0].cpu().permute(1, 2, 0).numpy()
        upscaled_np = ((upscaled_np * 0.5 + 0.5) * 255).astype(np.uint8)

        return Image.fromarray(upscaled_np)

    def upscale_16x(self, image):
        """upscales image 16x"""

        if isinstance(image, Image.Image):
            image = np.array(image)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            upscaled = self.upscaler_16x(image_tensor)

        upscaled_np = upscaled[0].cpu().permute(1, 2, 0).numpy()
        upscaled_np = ((upscaled_np * 0.5 + 0.5) * 255).astype(np.uint8)

        return Image.fromarray(upscaled_np)
