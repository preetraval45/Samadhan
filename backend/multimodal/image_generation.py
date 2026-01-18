"""
Image Generation Module
Supports multiple backends: Stable Diffusion, DALL-E style generation
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import base64

class ImageStyle(str, Enum):
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    DIGITAL_ART = "digital_art"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"

class ImageGenerator:
    """
    Custom image generation using Stable Diffusion XL as base
    """

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",  # Free, open-source model
        device: str = "cuda",
        use_custom_model: bool = True
    ):
        self.device = device
        self.model_path = model_path
        self.use_custom_lora = use_custom_lora
        self.pipeline = None

    async def initialize(self):
        """Load the model pipeline"""
        print(f"Loading image generation model from {self.model_path}...")

        # Load SDXL pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )

        # Optimize scheduler for faster generation
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        # Move to GPU
        if self.device == "cuda" and torch.cuda.is_available():
            self.pipeline = self.pipeline.to(self.device)
            # Enable memory optimizations
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()

        # Load custom LoRA adapters if available
        if self.use_custom_lora:
            await self._load_custom_lora()

        print("Image generation model loaded successfully")

    async def _load_custom_lora(self):
        """Load custom LoRA adapters for style customization"""
        # TODO: Implement custom LoRA loading
        # self.pipeline.load_lora_weights("./models/custom_lora")
        pass

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: ImageStyle = ImageStyle.REALISTIC,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate images from text prompt

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in generation
            style: Image style preset
            width: Output width (must be multiple of 8)
            height: Output height (must be multiple of 8)
            num_inference_steps: Quality vs speed tradeoff (20-50)
            guidance_scale: How closely to follow prompt (5-15)
            num_images: Number of variations to generate
            seed: Random seed for reproducibility

        Returns:
            List of generated images with metadata
        """
        if self.pipeline is None:
            await self.initialize()

        # Apply style-specific prompt engineering
        enhanced_prompt = self._apply_style_prompt(prompt, style)
        enhanced_negative = self._get_default_negative_prompt(negative_prompt, style)

        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate images
        result = self.pipeline(
            prompt=enhanced_prompt,
            negative_prompt=enhanced_negative,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator
        )

        # Convert to base64 for API response
        images_data = []
        for idx, image in enumerate(result.images):
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            images_data.append({
                "index": idx,
                "image": img_base64,
                "format": "png",
                "width": width,
                "height": height,
                "seed": seed,
                "prompt": enhanced_prompt,
                "style": style.value
            })

        return images_data

    def _apply_style_prompt(self, prompt: str, style: ImageStyle) -> str:
        """Apply style-specific prompt engineering"""
        style_prompts = {
            ImageStyle.REALISTIC: f"{prompt}, photorealistic, highly detailed, 8k uhd, dslr, soft lighting, high quality",
            ImageStyle.ARTISTIC: f"{prompt}, artistic, creative, masterpiece, trending on artstation",
            ImageStyle.ANIME: f"{prompt}, anime style, manga, cel shaded, vibrant colors, studio ghibli",
            ImageStyle.CINEMATIC: f"{prompt}, cinematic lighting, dramatic, film grain, depth of field, bokeh",
            ImageStyle.DIGITAL_ART: f"{prompt}, digital art, concept art, highly detailed, vibrant colors",
            ImageStyle.OIL_PAINTING: f"{prompt}, oil painting, classical art, brushstrokes, canvas texture",
            ImageStyle.WATERCOLOR: f"{prompt}, watercolor painting, soft edges, flowing colors, artistic",
            ImageStyle.SKETCH: f"{prompt}, pencil sketch, line art, detailed drawing, monochrome"
        }
        return style_prompts.get(style, prompt)

    def _get_default_negative_prompt(self, negative_prompt: Optional[str], style: ImageStyle) -> str:
        """Get default negative prompt based on style"""
        base_negative = "low quality, blurry, pixelated, distorted, ugly, duplicate, morbid, mutilated"

        if negative_prompt:
            return f"{negative_prompt}, {base_negative}"

        return base_negative

    async def generate_variations(
        self,
        image: Image.Image,
        prompt: str,
        num_variations: int = 4,
        strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generate variations of an existing image

        Args:
            image: Source image
            prompt: Description for variations
            num_variations: Number of variations
            strength: How much to change (0.0-1.0)
        """
        # TODO: Implement image-to-image generation
        # This requires img2img pipeline
        raise NotImplementedError("Image variations not yet implemented")

    async def upscale(
        self,
        image: Image.Image,
        scale_factor: int = 2
    ) -> Dict[str, Any]:
        """
        Upscale image using AI

        Args:
            image: Source image
            scale_factor: Upscaling factor (2x, 4x)
        """
        # TODO: Implement Real-ESRGAN or similar
        raise NotImplementedError("Image upscaling not yet implemented")

    def cleanup(self):
        """Free GPU memory"""
        if self.pipeline is not None:
            del self.pipeline
            torch.cuda.empty_cache()


# Singleton instance
_image_generator: Optional[ImageGenerator] = None

async def get_image_generator() -> ImageGenerator:
    """Get or create image generator instance"""
    global _image_generator
    if _image_generator is None:
        _image_generator = ImageGenerator()
        await _image_generator.initialize()
    return _image_generator
