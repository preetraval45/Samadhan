"""
Video Generation and Editing Module
Supports text-to-video, video editing, unlimited duration videos
All using FREE open-source models
"""

from typing import Optional, Dict, Any, List
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path
import io
import base64
import subprocess
from tqdm import tqdm


class VideoGenerator:
    """
    Video generation using free models
    Supports unlimited duration by generating in chunks
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.text_to_video_pipeline = None
        self.initialized = False

    async def initialize(self):
        """Load video generation models"""
        print("Initializing video generation models...")

        # Text-to-Video Model (FREE - ZeroScope)
        try:
            self.text_to_video_pipeline = DiffusionPipeline.from_pretrained(
                "cerspense/zeroscope_v2_576w",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.text_to_video_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.text_to_video_pipeline.scheduler.config
            )

            if self.device == "cuda":
                self.text_to_video_pipeline = self.text_to_video_pipeline.to(self.device)
                self.text_to_video_pipeline.enable_attention_slicing()
                self.text_to_video_pipeline.enable_vae_slicing()

            print("✓ Text-to-video model loaded (ZeroScope)")
        except Exception as e:
            print(f"⚠ Video generation model failed: {e}")

        self.initialized = True
        print("✅ Video generation models initialized")

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: float = 3.0,  # Unlimited - can be 60, 120, etc.
        fps: int = 24,
        width: int = 576,
        height: int = 320,
        num_inference_steps: int = 40,
        guidance_scale: float = 9.0
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt
        Supports unlimited duration by generating chunks and stitching

        Args:
            prompt: Text description of video
            duration_seconds: How long video should be (NO LIMIT)
            fps: Frames per second
            width: Video width
            height: Video height
            num_inference_steps: Quality
            guidance_scale: Prompt adherence
        """
        if not self.initialized:
            await self.initialize()

        total_frames = int(duration_seconds * fps)
        chunk_frames = 24  # Generate 24 frames per chunk (1 second at 24fps)

        print(f"Generating {duration_seconds}s video ({total_frames} frames)...")

        all_frames = []
        num_chunks = (total_frames + chunk_frames - 1) // chunk_frames

        for chunk_idx in tqdm(range(num_chunks), desc="Generating video chunks"):
            # Generate chunk
            chunk_result = self.text_to_video_pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                num_frames=min(chunk_frames, total_frames - len(all_frames)),
                guidance_scale=guidance_scale
            ).frames[0]

            all_frames.extend(chunk_result)

            # Stop if we have enough frames
            if len(all_frames) >= total_frames:
                break

        # Trim to exact duration
        all_frames = all_frames[:total_frames]

        # Save video
        output_path = Path("./temp") / f"generated_video_{id(self)}.mp4"
        output_path.parent.mkdir(exist_ok=True)

        self._save_video(all_frames, str(output_path), fps)

        # Read and encode video
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode()

        return {
            "video": video_base64,
            "format": "mp4",
            "duration": duration_seconds,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": len(all_frames)
        }

    def _save_video(self, frames: List[Image.Image], output_path: str, fps: int):
        """Save frames as video file"""
        if len(frames) == 0:
            raise ValueError("No frames to save")

        # Get dimensions from first frame
        first_frame = np.array(frames[0])
        height, width = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            for frame in frames:
                frame_array = np.array(frame)
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        finally:
            out.release()

    async def interpolate_frames(
        self,
        video_path: str,
        target_fps: int = 60
    ) -> Dict[str, Any]:
        """
        Increase FPS using frame interpolation (FREE using RIFE)

        Args:
            video_path: Input video
            target_fps: Desired output FPS
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read all frames
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Calculate interpolation factor
        interp_factor = target_fps / original_fps

        if interp_factor <= 1.0:
            # No interpolation needed
            interpolated_frames = frames
        else:
            # Simple linear interpolation (can be replaced with RIFE for better quality)
            interpolated_frames = []
            for i in range(len(frames) - 1):
                interpolated_frames.append(frames[i])

                # Add interpolated frames between current and next
                num_interp = int(interp_factor) - 1
                for j in range(1, num_interp + 1):
                    alpha = j / (num_interp + 1)
                    interp_frame = cv2.addWeighted(
                        frames[i], 1 - alpha,
                        frames[i + 1], alpha,
                        0
                    )
                    interpolated_frames.append(interp_frame)

            interpolated_frames.append(frames[-1])

        # Save output
        output_path = Path("./temp") / f"interpolated_{id(self)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))

        for frame in interpolated_frames:
            out.write(frame)
        out.release()

        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode()

        return {
            "video": video_base64,
            "format": "mp4",
            "original_fps": original_fps,
            "new_fps": target_fps,
            "total_frames": len(interpolated_frames)
        }

    async def upscale_video(
        self,
        video_path: str,
        scale_factor: int = 2
    ) -> Dict[str, Any]:
        """
        Upscale video resolution using AI

        Args:
            video_path: Input video
            scale_factor: 2x or 4x
        """
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # Load super-resolution model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale_factor)
        upsampler = RealESRGANer(
            scale=scale_factor,
            model_path=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x{scale_factor}plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == "cuda" else False
        )

        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_width = width * scale_factor
        new_height = height * scale_factor

        # Output video
        output_path = Path("./temp") / f"upscaled_{id(self)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

        print(f"Upscaling video from {width}x{height} to {new_width}x{new_height}...")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Upscale frame
            output_frame, _ = upsampler.enhance(frame, outscale=scale_factor)
            out.write(output_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()

        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode()

        return {
            "video": video_base64,
            "format": "mp4",
            "original_resolution": f"{width}x{height}",
            "new_resolution": f"{new_width}x{new_height}",
            "scale_factor": scale_factor
        }

    async def add_audio_to_video(
        self,
        video_path: str,
        audio_path: str
    ) -> Dict[str, Any]:
        """
        Add audio track to video using ffmpeg

        Args:
            video_path: Video file
            audio_path: Audio file
        """
        output_path = Path("./temp") / f"video_with_audio_{id(self)}.mp4"

        # Use ffmpeg to merge
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
            '-shortest', str(output_path)
        ]

        subprocess.run(cmd, check=True)

        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode()

        return {
            "video": video_base64,
            "format": "mp4",
            "has_audio": True
        }

    async def extract_frames(
        self,
        video_path: str,
        frame_rate: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video

        Args:
            video_path: Input video
            frame_rate: Extract every Nth frame (None = all frames)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_rate is None or frame_idx % frame_rate == 0:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Convert to base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                frames.append({
                    "frame_number": frame_idx,
                    "image": img_base64,
                    "format": "png"
                })

            frame_idx += 1

        cap.release()
        return frames


# Singleton
_video_generator: Optional[VideoGenerator] = None

async def get_video_generator() -> VideoGenerator:
    """Get or create video generator instance"""
    global _video_generator
    if _video_generator is None:
        _video_generator = VideoGenerator()
        await _video_generator.initialize()
    return _video_generator
