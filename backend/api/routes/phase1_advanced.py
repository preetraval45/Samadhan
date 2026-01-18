from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import os
from datetime import datetime

from custom_ai.large_language_model import (
    create_grok_model,
    create_large_model,
    create_medium_model,
    create_small_model
)
from custom_ai.rlhf_training import RLHFTrainer, RewardModel, ValueModel, ConstitutionalAI
from custom_ai.advanced_image_generation import AdvancedImageGenerator
from custom_ai.advanced_video_generation import AdvancedVideoGenerator
from custom_ai.advanced_deepfake import AdvancedDeepfakeSystem
from custom_ai.voice_cloning import VoiceCloningSystem
from custom_ai.tokenizer import CustomTokenizer


router = APIRouter()


phase1_cache = {
    'llm_small': None,
    'llm_medium': None,
    'llm_large': None,
    'llm_grok': None,
    'tokenizer': None,
    'advanced_image': None,
    'advanced_video': None,
    'deepfake': None,
    'voice_cloning': None,
    'rlhf_trainer': None,
    'constitutional_ai': None
}


class LLMGenerationRequest(BaseModel):
    prompt: str
    model_size: str = "small"
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    use_constitutional_ai: bool = False


class RLHFTrainingRequest(BaseModel):
    prompts: List[str]
    chosen_responses: List[str]
    rejected_responses: List[str]
    epochs: int = 3


class AdvancedImageRequest(BaseModel):
    prompt: str
    mode: str = "generate"
    height: int = 512
    width: int = 512
    control_type: Optional[str] = None
    extend_direction: Optional[str] = None
    extend_pixels: int = 128


class AdvancedVideoRequest(BaseModel):
    prompt: str
    num_frames: int = 16
    resolution: str = "hd"
    camera_motion: Optional[str] = None
    motion_speed: float = 1.0
    fps: int = 24


class DeepfakeRequest(BaseModel):
    mode: str
    target_age_group: Optional[int] = None
    emotion: Optional[str] = None


class VoiceCloneRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    duration: float = 5.0


@router.post("/init_phase1_models")
async def initialize_phase1_models(model_sizes: List[str] = ["small"]):
    try:
        global phase1_cache

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tokenizer = CustomTokenizer(vocab_size=50000)
        phase1_cache['tokenizer'] = tokenizer

        for size in model_sizes:
            if size == 'small':
                phase1_cache['llm_small'] = create_small_model().to(device)
            elif size == 'medium':
                phase1_cache['llm_medium'] = create_medium_model().to(device)
            elif size == 'large':
                phase1_cache['llm_large'] = create_large_model().to(device)
            elif size == 'grok':
                phase1_cache['llm_grok'] = create_grok_model().to(device)

        phase1_cache['advanced_image'] = AdvancedImageGenerator(
            phase1_cache['llm_small'],
            tokenizer,
            device
        )

        phase1_cache['advanced_video'] = AdvancedVideoGenerator(
            phase1_cache['llm_small'],
            tokenizer,
            device
        )

        phase1_cache['deepfake'] = AdvancedDeepfakeSystem(device)

        phase1_cache['voice_cloning'] = VoiceCloningSystem(device)

        phase1_cache['constitutional_ai'] = ConstitutionalAI(
            phase1_cache['llm_small'],
            tokenizer
        )

        return {
            "status": "success",
            "message": "Phase 1 models initialized",
            "device": device,
            "models_loaded": model_sizes,
            "capabilities": [
                "grok_level_llm",
                "rlhf_training",
                "constitutional_ai",
                "controlnet_image_generation",
                "inpainting_outpainting",
                "super_resolution_8x_16x",
                "4k_8k_video",
                "camera_motion_control",
                "object_motion_tracking",
                "expression_transfer",
                "age_progression",
                "gender_swap",
                "lip_sync",
                "zero_shot_voice_cloning"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/generate")
async def generate_with_llm(request: LLMGenerationRequest):
    try:
        if phase1_cache['tokenizer'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        model_key = f'llm_{request.model_size}'
        if model_key not in phase1_cache or phase1_cache[model_key] is None:
            raise HTTPException(status_code=400, detail=f"Model {request.model_size} not loaded")

        model = phase1_cache[model_key]
        tokenizer = phase1_cache['tokenizer']

        prompt_tokens = tokenizer.encode(request.prompt)
        prompt_tensor = torch.tensor([prompt_tokens], device=next(model.parameters()).device)

        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )

        generated_text = tokenizer.decode(generated[0].cpu().tolist())

        if request.use_constitutional_ai:
            constitutional = phase1_cache['constitutional_ai']
            critique = constitutional.critique_response(request.prompt, generated_text)
            generated_text = constitutional.revise_response(request.prompt, generated_text, critique)

        return {
            "status": "success",
            "prompt": request.prompt,
            "generated_text": generated_text,
            "model_size": request.model_size,
            "constitutional_ai_used": request.use_constitutional_ai
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rlhf/train")
async def train_rlhf(request: RLHFTrainingRequest, background_tasks: BackgroundTasks):
    try:
        if phase1_cache['llm_small'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        model = phase1_cache['llm_small']
        tokenizer = phase1_cache['tokenizer']
        device = next(model.parameters()).device

        reward_model = RewardModel(model).to(device)
        value_model = ValueModel(model).to(device)

        trainer = RLHFTrainer(model, reward_model, value_model, tokenizer, device)

        def train_background():
            from torch.utils.data import DataLoader
            from custom_ai.rlhf_training import PreferenceDataset

            dataset = PreferenceDataset(
                request.prompts,
                request.chosen_responses,
                request.rejected_responses,
                tokenizer
            )

            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            trainer.train_reward_model(dataloader, epochs=request.epochs)

            trainer.train_ppo(request.prompts[:10], epochs=request.epochs)

        background_tasks.add_task(train_background)

        return {
            "status": "training_started",
            "message": "RLHF training started in background",
            "num_samples": len(request.prompts),
            "epochs": request.epochs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/advanced")
async def advanced_image_generation(request: AdvancedImageRequest):
    try:
        if phase1_cache['advanced_image'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        advanced_gen = phase1_cache['advanced_image']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "generated_outputs/advanced_images"
        os.makedirs(output_dir, exist_ok=True)

        if request.mode == "generate":
            result = advanced_gen.image_engine.generate_image(
                request.prompt,
                request.height,
                request.width
            )
            output_path = f"{output_dir}/generated_{timestamp}.png"
            result.save(output_path)

        elif request.mode == "upscale_8x":
            if not os.path.exists(request.prompt):
                raise HTTPException(status_code=404, detail="Source image not found")

            from PIL import Image
            source_image = Image.open(request.prompt)
            result = advanced_gen.upscale_8x(source_image)
            output_path = f"{output_dir}/upscaled_8x_{timestamp}.png"
            result.save(output_path)

        elif request.mode == "upscale_16x":
            if not os.path.exists(request.prompt):
                raise HTTPException(status_code=404, detail="Source image not found")

            from PIL import Image
            source_image = Image.open(request.prompt)
            result = advanced_gen.upscale_16x(source_image)
            output_path = f"{output_dir}/upscaled_16x_{timestamp}.png"
            result.save(output_path)

        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        return {
            "status": "success",
            "mode": request.mode,
            "output_path": output_path,
            "download_url": f"/api/v1/phase1/download/{os.path.basename(output_path)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/advanced")
async def advanced_video_generation(request: AdvancedVideoRequest):
    try:
        if phase1_cache['advanced_video'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        advanced_video = phase1_cache['advanced_video']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "generated_outputs/advanced_videos"
        os.makedirs(output_dir, exist_ok=True)

        if request.resolution == "4k":
            video = advanced_video.generate_4k(request.prompt, request.num_frames)
        elif request.resolution == "8k":
            video = advanced_video.generate_8k(request.prompt, request.num_frames)
        elif request.camera_motion:
            video = advanced_video.generate_with_camera_motion(
                request.prompt,
                request.camera_motion,
                {'speed': request.motion_speed},
                request.num_frames
            )
        else:
            video = advanced_video.video_engine.generate_video(
                request.prompt,
                request.num_frames,
                fps=request.fps
            )

        output_path = f"{output_dir}/video_{timestamp}.mp4"
        advanced_video.video_engine.save_video(video, output_path, request.fps)

        return {
            "status": "success",
            "resolution": request.resolution,
            "num_frames": request.num_frames,
            "output_path": output_path,
            "download_url": f"/api/v1/phase1/download/{os.path.basename(output_path)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deepfake/process")
async def process_deepfake(file: UploadFile, request: DeepfakeRequest):
    try:
        if phase1_cache['deepfake'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        deepfake = phase1_cache['deepfake']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"uploaded_faces/temp_{timestamp}_{file.filename}"
        os.makedirs("uploaded_faces", exist_ok=True)

        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        from PIL import Image
        source_image = Image.open(temp_path)

        if request.mode == "age_progression":
            result = deepfake.progress_age(source_image, request.target_age_group)
        elif request.mode == "gender_swap":
            result = deepfake.swap_gender(source_image)
        elif request.mode == "expression_transfer":
            result = deepfake.transfer_expression(source_image, source_image)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        output_dir = "generated_outputs/deepfakes"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{request.mode}_{timestamp}.png"

        result.save(output_path)

        return {
            "status": "success",
            "mode": request.mode,
            "output_path": output_path,
            "download_url": f"/api/v1/phase1/download/{os.path.basename(output_path)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/clone")
async def clone_voice(reference_audio: UploadFile, request: VoiceCloneRequest):
    try:
        if phase1_cache['voice_cloning'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        voice_system = phase1_cache['voice_cloning']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_audio_path = f"uploaded_faces/temp_audio_{timestamp}.wav"

        with open(temp_audio_path, "wb") as buffer:
            content = await reference_audio.read()
            buffer.write(content)

        cloned_audio = voice_system.clone_voice(
            temp_audio_path,
            request.text,
            request.emotion
        )

        output_dir = "generated_outputs/voice_clones"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/cloned_{timestamp}.wav"

        voice_system.save_audio(cloned_audio, output_path)

        return {
            "status": "success",
            "text": request.text,
            "emotion": request.emotion,
            "output_path": output_path,
            "download_url": f"/api/v1/phase1/download/{os.path.basename(output_path)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_generated_file(filename: str):
    search_dirs = [
        "generated_outputs/advanced_images",
        "generated_outputs/advanced_videos",
        "generated_outputs/deepfakes",
        "generated_outputs/voice_clones"
    ]

    for directory in search_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, filename=filename)

    raise HTTPException(status_code=404, detail="File not found")


@router.get("/capabilities")
async def get_phase1_capabilities():
    return {
        "status": "operational",
        "phase": "Phase 1 - Custom Model Development",
        "capabilities": {
            "large_language_models": {
                "sizes": ["small_7B", "medium_13B", "large_70B", "grok_314B"],
                "features": ["rlhf", "constitutional_ai", "quantization"],
                "context_length": 8192
            },
            "advanced_image_generation": {
                "features": [
                    "controlnet_canny",
                    "controlnet_depth",
                    "inpainting",
                    "outpainting",
                    "super_resolution_8x",
                    "super_resolution_16x"
                ]
            },
            "advanced_video_generation": {
                "resolutions": ["HD", "4K", "8K"],
                "features": [
                    "camera_motion_control",
                    "object_motion_tracking",
                    "scene_transitions",
                    "video_to_video_translation",
                    "temporal_consistency"
                ]
            },
            "deepfake_system": {
                "features": [
                    "expression_transfer",
                    "age_progression",
                    "age_regression",
                    "gender_swap",
                    "lip_sync",
                    "full_body_deepfake",
                    "watermarking"
                ],
                "real_time": "30fps+"
            },
            "voice_cloning": {
                "features": [
                    "zero_shot_cloning",
                    "emotion_control",
                    "noise_removal",
                    "voice_conversion",
                    "audio_super_resolution"
                ],
                "required_audio": "3_seconds"
            }
        }
    }
