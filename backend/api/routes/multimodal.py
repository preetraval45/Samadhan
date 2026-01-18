"""
Multimodal API Routes
Unified endpoints for all AI generation capabilities
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import base64
from PIL import Image
import io
import numpy as np

from multimodal.image_generation import get_image_generator, ImageStyle
from multimodal.image_editing import get_image_editor
from multimodal.deepfake_engine import get_deepfake_engine, DeepfakeConfig
from multimodal.video_generation import get_video_generator
from multimodal.audio_generation import get_audio_generator
from multimodal.model_3d_generation import get_model_3d_generator

router = APIRouter(prefix="/api/v1/multimodal", tags=["multimodal"])


# ============== Image Generation ==============

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of image")
    negative_prompt: Optional[str] = Field(None, description="What to avoid")
    style: ImageStyle = Field(ImageStyle.REALISTIC, description="Image style")
    width: int = Field(1024, ge=512, le=2048, description="Width (must be multiple of 8)")
    height: int = Field(1024, ge=512, le=2048, description="Height (must be multiple of 8)")
    num_inference_steps: int = Field(30, ge=20, le=100, description="Quality steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Prompt adherence")
    num_images: int = Field(1, ge=1, le=4, description="Number of images")
    seed: Optional[int] = Field(None, description="Random seed")


@router.post("/generate/image")
async def generate_image(request: ImageGenerationRequest):
    """Generate images from text prompt"""
    try:
        generator = await get_image_generator()
        result = await generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            style=request.style,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=request.seed
        )
        return {"success": True, "images": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Image Editing ==============

@router.post("/edit/image/inpaint")
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None)
):
    """Fill masked area with AI-generated content"""
    try:
        # Load images
        img = Image.open(io.BytesIO(await image.read()))
        mask_img = Image.open(io.BytesIO(await mask.read()))

        editor = await get_image_editor()
        result = await editor.inpaint(img, mask_img, prompt, negative_prompt)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit/image/upscale")
async def upscale_image(
    image: UploadFile = File(...),
    scale_factor: int = Form(4, ge=2, le=4)
):
    """Upscale image using AI"""
    try:
        img = Image.open(io.BytesIO(await image.read()))
        editor = await get_image_editor()
        result = await editor.upscale(img, scale_factor)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit/image/remove_background")
async def remove_background(
    image: UploadFile = File(...),
    alpha_matting: bool = Form(True)
):
    """Remove background from image"""
    try:
        img = Image.open(io.BytesIO(await image.read()))
        editor = await get_image_editor()
        result = await editor.remove_background(img, alpha_matting)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit/image/face_restore")
async def restore_face(image: UploadFile = File(...)):
    """Restore/enhance faces in image"""
    try:
        img = Image.open(io.BytesIO(await image.read()))
        editor = await get_image_editor()
        result = await editor.face_restoration(img)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Deepfake ==============

@router.post("/deepfake/face_swap")
async def face_swap(
    source_image: UploadFile = File(..., description="Face to use"),
    target_image: UploadFile = File(..., description="Where to place face"),
    face_enhancement: bool = Form(True),
    super_resolution: bool = Form(True)
):
    """Swap faces between images"""
    try:
        # Load images
        source = np.array(Image.open(io.BytesIO(await source_image.read())))
        target = np.array(Image.open(io.BytesIO(await target_image.read())))

        # Convert RGB to BGR
        source = source[:, :, ::-1]
        target = target[:, :, ::-1]

        config = DeepfakeConfig(
            face_enhancement=face_enhancement,
            super_resolution=super_resolution
        )

        engine = await get_deepfake_engine()
        result_img, metadata = await engine.swap_face(source, target, config)

        # Convert back to RGB and encode
        result_img = result_img[:, :, ::-1]
        pil_img = Image.fromarray(result_img)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "image": img_base64,
            "format": "png",
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Video Generation ==============

class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of video")
    duration_seconds: float = Field(3.0, ge=1.0, description="Duration (UNLIMITED)")
    fps: int = Field(24, ge=12, le=60, description="Frames per second")
    width: int = Field(576, ge=256, le=1024)
    height: int = Field(320, ge=256, le=1024)
    num_inference_steps: int = Field(40, ge=20, le=100)
    guidance_scale: float = Field(9.0, ge=1.0, le=20.0)


@router.post("/generate/video")
async def generate_video(request: VideoGenerationRequest):
    """Generate video from text prompt - UNLIMITED DURATION"""
    try:
        generator = await get_video_generator()
        result = await generator.generate_video(
            prompt=request.prompt,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Audio Generation ==============

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: str = Field("en", description="Language code")
    speaker: Optional[str] = Field(None, description="Speaker voice")


@router.post("/generate/audio/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech"""
    try:
        generator = await get_audio_generator()
        result = await generator.text_to_speech(
            text=request.text,
            language=request.language,
            speaker=request.speaker
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/audio/voice_clone")
async def clone_voice(
    reference_audio: UploadFile = File(...),
    text: str = Form(...)
):
    """Clone voice from audio sample"""
    try:
        # Save reference audio temporarily
        from pathlib import Path
        temp_path = Path("./temp") / f"ref_audio_{id(reference_audio)}.wav"
        temp_path.parent.mkdir(exist_ok=True)

        with open(temp_path, 'wb') as f:
            f.write(await reference_audio.read())

        generator = await get_audio_generator()
        result = await generator.clone_voice(str(temp_path), text)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== 3D Generation ==============

class Model3DRequest(BaseModel):
    prompt: str = Field(..., description="Text description of 3D model")
    output_format: str = Field("glb", description="Output format (glb, obj, ply, stl)")
    guidance_scale: float = Field(15.0, ge=1.0, le=20.0)


@router.post("/generate/3d")
async def generate_3d_model(request: Model3DRequest):
    """Generate 3D model from text"""
    try:
        generator = await get_model_3d_generator()
        result = await generator.generate_3d_model(
            prompt=request.prompt,
            output_format=request.output_format,
            guidance_scale=request.guidance_scale
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Model Info ==============

@router.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    from backend.multimodal.free_models_config import list_available_models

    models = list_available_models()
    return {
        "success": True,
        "models": models,
        "all_free": True,
        "unlimited_usage": True
    }
