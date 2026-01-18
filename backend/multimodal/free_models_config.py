"""
Configuration for Free and Open-Source AI Models
All models listed here are completely free to use
"""

from typing import Dict, Any

# Free Image Generation Models
FREE_IMAGE_MODELS = {
    "stable_diffusion_v1_5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "type": "text-to-image",
        "license": "CreativeML Open RAIL-M",
        "description": "Free Stable Diffusion 1.5",
        "requires_gpu": True,
        "vram_required": "4GB+"
    },
    "stable_diffusion_v2_1": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "type": "text-to-image",
        "license": "CreativeML Open RAIL++-M",
        "description": "Free Stable Diffusion 2.1",
        "requires_gpu": True,
        "vram_required": "6GB+"
    },
    "dreamlike_photoreal": {
        "model_id": "dreamlike-art/dreamlike-photoreal-2.0",
        "type": "text-to-image",
        "license": "CreativeML Open RAIL-M",
        "description": "Free photorealistic model",
        "requires_gpu": True,
        "vram_required": "4GB+"
    },
    "openjourney": {
        "model_id": "prompthero/openjourney",
        "type": "text-to-image",
        "license": "CreativeML Open RAIL-M",
        "description": "Free Midjourney-style model",
        "requires_gpu": True,
        "vram_required": "4GB+"
    }
}

# Free Image Editing Models
FREE_IMAGE_EDITING_MODELS = {
    "controlnet": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "type": "image-control",
        "license": "Apache 2.0",
        "description": "Free ControlNet for guided generation",
        "requires_gpu": True
    },
    "instruct_pix2pix": {
        "model_id": "timbrooks/instruct-pix2pix",
        "type": "image-editing",
        "license": "MIT",
        "description": "Free instruction-based editing",
        "requires_gpu": True
    }
}

# Free Text-to-Speech Models
FREE_TTS_MODELS = {
    "coqui_tts": {
        "model_id": "coqui/XTTS-v2",
        "type": "text-to-speech",
        "license": "CPML",
        "description": "Free multilingual TTS",
        "requires_gpu": False
    },
    "piper_tts": {
        "model_id": "rhasspy/piper",
        "type": "text-to-speech",
        "license": "MIT",
        "description": "Free fast TTS",
        "requires_gpu": False
    }
}

# Free Speech-to-Text Models
FREE_STT_MODELS = {
    "whisper_large": {
        "model_id": "openai/whisper-large-v3",
        "type": "speech-to-text",
        "license": "MIT",
        "description": "Free Whisper STT (open-source)",
        "requires_gpu": True
    },
    "whisper_medium": {
        "model_id": "openai/whisper-medium",
        "type": "speech-to-text",
        "license": "MIT",
        "description": "Free Whisper medium model",
        "requires_gpu": False
    }
}

# Free Video Models
FREE_VIDEO_MODELS = {
    "zeroscope": {
        "model_id": "cerspense/zeroscope_v2_576w",
        "type": "text-to-video",
        "license": "CreativeML Open RAIL++-M",
        "description": "Free text-to-video",
        "requires_gpu": True,
        "vram_required": "8GB+"
    }
}

# Free 3D Models
FREE_3D_MODELS = {
    "shap_e": {
        "model_id": "openai/shap-e",
        "type": "text-to-3d",
        "license": "MIT",
        "description": "Free 3D generation (open-source)",
        "requires_gpu": True
    }
}

# Free LLM Models (for chat and reasoning)
FREE_LLM_MODELS = {
    "llama_3_1_8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "type": "text-generation",
        "license": "Llama 3.1 Community License",
        "description": "Free Llama 3.1 8B",
        "requires_gpu": True,
        "vram_required": "16GB+"
    },
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "type": "text-generation",
        "license": "Apache 2.0",
        "description": "Free Mistral 7B",
        "requires_gpu": True,
        "vram_required": "14GB+"
    },
    "phi_3_mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "type": "text-generation",
        "license": "MIT",
        "description": "Free Phi-3 (small but powerful)",
        "requires_gpu": False,
        "vram_required": "4GB+"
    }
}

# Custom Model Training Configurations
CUSTOM_MODEL_TRAINING = {
    "image_generation": {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "training_method": "DreamBooth + LoRA",
        "dataset_size": "50-100 images minimum",
        "training_time": "2-4 hours on GPU",
        "hardware": "GPU with 12GB+ VRAM recommended",
        "frameworks": ["diffusers", "transformers", "peft"]
    },
    "voice_cloning": {
        "base_model": "coqui/XTTS-v2",
        "training_method": "Fine-tuning",
        "dataset_size": "5-10 minutes of clean audio",
        "training_time": "30 minutes - 2 hours",
        "hardware": "CPU or GPU",
        "frameworks": ["TTS", "torch"]
    },
    "face_swap": {
        "base_model": "custom",
        "training_method": "Train from scratch",
        "dataset_size": "500+ face images per person",
        "training_time": "12-24 hours",
        "hardware": "GPU with 8GB+ VRAM",
        "frameworks": ["insightface", "onnxruntime", "opencv"]
    },
    "video_generation": {
        "base_model": "cerspense/zeroscope_v2_576w",
        "training_method": "Fine-tuning",
        "dataset_size": "100+ video clips",
        "training_time": "24-48 hours",
        "hardware": "GPU with 16GB+ VRAM",
        "frameworks": ["diffusers", "torch"]
    }
}

def get_model_config(model_type: str, model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    model_configs = {
        "image_generation": FREE_IMAGE_MODELS,
        "image_editing": FREE_IMAGE_EDITING_MODELS,
        "tts": FREE_TTS_MODELS,
        "stt": FREE_STT_MODELS,
        "video": FREE_VIDEO_MODELS,
        "3d": FREE_3D_MODELS,
        "llm": FREE_LLM_MODELS
    }

    return model_configs.get(model_type, {}).get(model_name, {})

def list_available_models(model_type: str = None) -> Dict[str, Any]:
    """List all available free models"""
    all_models = {
        "image_generation": FREE_IMAGE_MODELS,
        "image_editing": FREE_IMAGE_EDITING_MODELS,
        "text_to_speech": FREE_TTS_MODELS,
        "speech_to_text": FREE_STT_MODELS,
        "video_generation": FREE_VIDEO_MODELS,
        "3d_generation": FREE_3D_MODELS,
        "language_models": FREE_LLM_MODELS
    }

    if model_type:
        return {model_type: all_models.get(model_type, {})}

    return all_models
