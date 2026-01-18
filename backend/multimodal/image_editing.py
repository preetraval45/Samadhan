"""
Image Editing Module
Supports inpainting, upscaling, background removal, style transfer, etc.
All using FREE open-source models
"""

from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from controlnet_aux import CannyDetector, HEDdetector
from rembg import remove as remove_background
import io
import base64


class ImageEditor:
    """Comprehensive image editing using free models"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.inpaint_pipeline = None
        self.controlnet_pipeline = None
        self.super_resolution = None
        self.initialized = False

    async def initialize(self):
        """Load all editing models"""
        print("Initializing image editing models...")

        # 1. Inpainting Model (FREE - Stable Diffusion)
        try:
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            if self.device == "cuda":
                self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
                self.inpaint_pipeline.enable_attention_slicing()
            print("✓ Inpainting model loaded")
        except Exception as e:
            print(f"⚠ Inpainting model failed: {e}")

        # 2. ControlNet for guided editing (FREE)
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            if self.device == "cuda":
                self.controlnet_pipeline = self.controlnet_pipeline.to(self.device)
                self.controlnet_pipeline.enable_attention_slicing()
            print("✓ ControlNet model loaded")
        except Exception as e:
            print(f"⚠ ControlNet model failed: {e}")

        # 3. Super Resolution (FREE - Real-ESRGAN)
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.super_resolution = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False
            )
            print("✓ Super-resolution model loaded")
        except Exception as e:
            print(f"⚠ Super-resolution model failed: {e}")

        self.initialized = True
        print("✅ Image editing models initialized")

    async def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """
        Fill masked area with AI-generated content

        Args:
            image: Original image
            mask: Mask (white = inpaint, black = keep)
            prompt: What to generate in masked area
            negative_prompt: What to avoid
            num_inference_steps: Quality (higher = better)
            guidance_scale: Prompt adherence
        """
        if not self.initialized:
            await self.initialize()

        if self.inpaint_pipeline is None:
            raise RuntimeError("Inpainting model not available")

        # Ensure correct sizes
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        # Generate inpainted image
        result = self.inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "blurry, low quality",
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        # Convert to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_base64,
            "format": "png",
            "width": result.width,
            "height": result.height
        }

    async def upscale(
        self,
        image: Image.Image,
        scale_factor: int = 4
    ) -> Dict[str, Any]:
        """
        Upscale image using AI super-resolution

        Args:
            image: Input image
            scale_factor: 2x or 4x
        """
        if not self.initialized:
            await self.initialize()

        if self.super_resolution is None:
            # Fallback to basic upscaling
            new_size = (image.width * scale_factor, image.height * scale_factor)
            result = image.resize(new_size, Image.LANCZOS)
        else:
            # AI upscaling
            img_array = np.array(image)
            if img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            output, _ = self.super_resolution.enhance(img_array, outscale=scale_factor)
            result = Image.fromarray(output)

        # Convert to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_base64,
            "format": "png",
            "width": result.width,
            "height": result.height,
            "scale_factor": scale_factor
        }

    async def remove_background(
        self,
        image: Image.Image,
        alpha_matting: bool = True
    ) -> Dict[str, Any]:
        """
        Remove background from image (FREE using rembg)

        Args:
            image: Input image
            alpha_matting: Better edge quality
        """
        # Use rembg (FREE, no API key)
        result = remove_background(image, alpha_matting=alpha_matting)

        # Convert to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_base64,
            "format": "png",
            "width": result.width,
            "height": result.height,
            "has_alpha": True
        }

    async def change_background(
        self,
        image: Image.Image,
        new_background: str,
        background_type: str = "color"
    ) -> Dict[str, Any]:
        """
        Replace background with color or image

        Args:
            image: Input image
            new_background: Hex color or image path
            background_type: 'color' or 'image'
        """
        # First remove background
        no_bg = remove_background(image)

        if background_type == "color":
            # Create solid color background
            bg = Image.new('RGB', no_bg.size, new_background)
            bg.paste(no_bg, (0, 0), no_bg)
            result = bg
        else:
            # Load background image
            bg = Image.open(new_background).resize(no_bg.size)
            bg.paste(no_bg, (0, 0), no_bg)
            result = bg

        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_base64,
            "format": "png",
            "width": result.width,
            "height": result.height
        }

    async def edge_detection(
        self,
        image: Image.Image,
        detector_type: str = "canny"
    ) -> np.ndarray:
        """
        Detect edges in image

        Args:
            image: Input image
            detector_type: 'canny' or 'hed'
        """
        img_array = np.array(image)

        if detector_type == "canny":
            detector = CannyDetector()
            edges = detector(img_array)
        else:
            detector = HEDdetector.from_pretrained('lllyasviel/Annotators')
            edges = detector(img_array)

        return edges

    async def guided_edit(
        self,
        image: Image.Image,
        prompt: str,
        control_type: str = "canny",
        num_inference_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Edit image with structure guidance (ControlNet)

        Args:
            image: Input image
            prompt: What to generate
            control_type: Type of control ('canny', 'hed', etc.)
            num_inference_steps: Quality
        """
        if not self.initialized:
            await self.initialize()

        # Detect edges
        edges = await self.edge_detection(image, control_type)

        # Generate using ControlNet
        result = self.controlnet_pipeline(
            prompt=prompt,
            image=Image.fromarray(edges),
            num_inference_steps=num_inference_steps
        ).images[0]

        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_base64,
            "format": "png",
            "width": result.width,
            "height": result.height
        }

    async def color_correction(
        self,
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0
    ) -> Dict[str, Any]:
        """
        Adjust image colors

        Args:
            image: Input image
            brightness: 0.5 to 2.0
            contrast: 0.5 to 2.0
            saturation: 0.0 to 2.0
        """
        from PIL import ImageEnhance

        # Apply adjustments
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_base64,
            "format": "png",
            "width": image.width,
            "height": image.height
        }

    async def face_restoration(
        self,
        image: Image.Image
    ) -> Dict[str, Any]:
        """
        Restore/enhance faces in image using GFPGAN
        """
        try:
            from gfpgan import GFPGANer

            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2
            )

            img_array = np.array(image)
            _, _, output = face_enhancer.enhance(
                img_array,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            result = Image.fromarray(output)

            buffered = io.BytesIO()
            result.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return {
                "image": img_base64,
                "format": "png",
                "width": result.width,
                "height": result.height
            }
        except Exception as e:
            raise RuntimeError(f"Face restoration failed: {e}")


# Singleton
_image_editor: Optional[ImageEditor] = None

async def get_image_editor() -> ImageEditor:
    """Get or create image editor instance"""
    global _image_editor
    if _image_editor is None:
        _image_editor = ImageEditor()
        await _image_editor.initialize()
    return _image_editor
