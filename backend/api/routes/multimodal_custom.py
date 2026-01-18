from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import os
import shutil
from datetime import datetime
import io

from custom_ai import CustomTokenizer
from custom_ai.image_generator import CustomImageGenerator, ImageGeneratorInference
from custom_ai.video_generator import CustomVideoGenerator, VideoGeneratorInference
from custom_ai.audio_generator import CustomAudioGenerator, AudioGeneratorInference
from custom_ai.face_preserving_generator import FacePreservingGenerator, FacePreservingInference
from custom_ai.multimodal_understanding import MultiModalUnderstanding, MultiModalQA


router = APIRouter()


multimodal_cache = {
    'tokenizer': None,
    'image_model': None,
    'video_model': None,
    'audio_model': None,
    'face_model': None,
    'understanding_model': None,
    'image_inference': None,
    'video_inference': None,
    'audio_inference': None,
    'face_inference': None,
    'multimodal_qa': None
}


os.makedirs('generated_outputs', exist_ok=True)
os.makedirs('uploaded_faces', exist_ok=True)


class ImageGenerationRequest(BaseModel):
    prompt: str
    height: int = 512
    width: int = 512
    num_steps: int = 50
    preserve_face: bool = False
    face_image_id: Optional[str] = None


class VideoGenerationRequest(BaseModel):
    prompt: str
    num_frames: int = 16
    height: int = 256
    width: int = 256
    fps: int = 8
    output_format: str = "mp4"
    preserve_face: bool = False
    face_image_id: Optional[str] = None
    long_video: bool = False
    total_frames: Optional[int] = 120


class AudioGenerationRequest(BaseModel):
    prompt: str
    duration: float = 5.0
    mode: str = "neural"


class QuestionRequest(BaseModel):
    question: str
    image_id: Optional[str] = None
    video_id: Optional[str] = None
    audio_id: Optional[str] = None


@router.post("/init_multimodal")
async def initialize_multimodal():
    try:
        global multimodal_cache

        vocab_size = 30000
        tokenizer = CustomTokenizer(vocab_size=vocab_size)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        image_model = CustomImageGenerator(vocab_size)
        video_model = CustomVideoGenerator(vocab_size)
        audio_model = CustomAudioGenerator(vocab_size)
        face_model = FacePreservingGenerator(vocab_size)
        understanding_model = MultiModalUnderstanding(vocab_size)

        image_inference = ImageGeneratorInference(image_model, tokenizer, device)
        video_inference = VideoGeneratorInference(video_model, tokenizer, device)
        audio_inference = AudioGeneratorInference(audio_model, tokenizer, device)
        face_inference = FacePreservingInference(face_model, tokenizer, device)
        multimodal_qa = MultiModalQA(understanding_model, tokenizer, device)

        multimodal_cache.update({
            'tokenizer': tokenizer,
            'image_model': image_model,
            'video_model': video_model,
            'audio_model': audio_model,
            'face_model': face_model,
            'understanding_model': understanding_model,
            'image_inference': image_inference,
            'video_inference': video_inference,
            'audio_inference': audio_inference,
            'face_inference': face_inference,
            'multimodal_qa': multimodal_qa
        })

        return {
            "status": "success",
            "message": "Multimodal AI initialized",
            "device": device,
            "capabilities": [
                "image_generation",
                "video_generation",
                "audio_generation",
                "face_preservation",
                "visual_question_answering",
                "multimodal_understanding"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_face")
async def upload_face(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_id = f"face_{timestamp}"
        file_path = f"uploaded_faces/{face_id}.jpg"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "face_id": face_id,
            "message": "Face uploaded successfully. Use this ID for face-preserving generation."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_image")
async def generate_image(request: ImageGenerationRequest):
    try:
        if multimodal_cache['image_inference'] is None and multimodal_cache['face_inference'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_outputs/image_{timestamp}.png"

        if request.preserve_face and request.face_image_id:
            face_path = f"uploaded_faces/{request.face_image_id}.jpg"

            if not os.path.exists(face_path):
                raise HTTPException(status_code=404, detail="Face image not found")

            image = multimodal_cache['face_inference'].generate_with_face(
                request.prompt,
                face_path,
                preserve_face=True
            )
        else:
            image = multimodal_cache['image_inference'].generate_image(
                request.prompt,
                request.height,
                request.width,
                request.num_steps
            )

        image.save(output_path)

        return {
            "status": "success",
            "prompt": request.prompt,
            "output_path": output_path,
            "download_url": f"/api/v1/multimodal/download/{os.path.basename(output_path)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_video")
async def generate_video(request: VideoGenerationRequest):
    try:
        if multimodal_cache['video_inference'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.long_video:
            video = multimodal_cache['video_inference'].generate_long_video(
                request.prompt,
                total_frames=request.total_frames or 120
            )
        else:
            if request.preserve_face and request.face_image_id:
                face_path = f"uploaded_faces/{request.face_image_id}.jpg"

                if not os.path.exists(face_path):
                    raise HTTPException(status_code=404, detail="Face image not found")

                video = multimodal_cache['face_inference'].generate_video_with_face(
                    request.prompt,
                    face_path,
                    num_frames=request.num_frames,
                    preserve_face=True
                )
            else:
                video = multimodal_cache['video_inference'].generate_video(
                    request.prompt,
                    request.num_frames,
                    request.height,
                    request.width,
                    request.fps
                )

        if request.output_format == "gif":
            output_path = f"generated_outputs/video_{timestamp}.gif"
            multimodal_cache['video_inference'].save_gif(video, output_path, request.fps)
        else:
            output_path = f"generated_outputs/video_{timestamp}.mp4"
            multimodal_cache['video_inference'].save_video(video, output_path, request.fps)

        return {
            "status": "success",
            "prompt": request.prompt,
            "num_frames": len(video),
            "output_path": output_path,
            "format": request.output_format,
            "download_url": f"/api/v1/multimodal/download/{os.path.basename(output_path)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_audio")
async def generate_audio(request: AudioGenerationRequest):
    try:
        if multimodal_cache['audio_inference'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_outputs/audio_{timestamp}.wav"

        audio = multimodal_cache['audio_inference'].generate_audio(
            request.prompt,
            request.duration,
            request.mode
        )

        multimodal_cache['audio_inference'].save_audio(audio, output_path)

        return {
            "status": "success",
            "prompt": request.prompt,
            "duration": request.duration,
            "mode": request.mode,
            "output_path": output_path,
            "download_url": f"/api/v1/multimodal/download/{os.path.basename(output_path)}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask_question")
async def ask_question(request: QuestionRequest):
    try:
        if multimodal_cache['multimodal_qa'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        image_path = None
        video_path = None
        audio_path = None

        if request.image_id:
            image_path = f"generated_outputs/{request.image_id}"
            if not os.path.exists(image_path):
                uploaded_path = f"uploaded_faces/{request.image_id}.jpg"
                if os.path.exists(uploaded_path):
                    image_path = uploaded_path
                else:
                    raise HTTPException(status_code=404, detail="Image not found")

        if request.video_id:
            video_path = f"generated_outputs/{request.video_id}"
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail="Video not found")

        if request.audio_id:
            audio_path = f"generated_outputs/{request.audio_id}"
            if not os.path.exists(audio_path):
                raise HTTPException(status_code=404, detail="Audio not found")

        answer = multimodal_cache['multimodal_qa'].answer(
            request.question,
            image_path=image_path,
            video_path=video_path,
            audio_path=audio_path
        )

        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/understand_content")
async def understand_content(file: UploadFile = File(...)):
    try:
        if multimodal_cache['multimodal_qa'] is None:
            raise HTTPException(status_code=400, detail="Models not initialized")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"generated_outputs/temp_{timestamp}_{file.filename}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        description = multimodal_cache['multimodal_qa'].understand_content(temp_path)

        return {
            "status": "success",
            "filename": file.filename,
            "description": description
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"generated_outputs/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


@router.get("/list_generated")
async def list_generated():
    try:
        files = os.listdir('generated_outputs')

        file_info = []
        for f in files:
            file_path = f"generated_outputs/{f}"
            size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)

            file_info.append({
                "filename": f,
                "size_bytes": size,
                "modified_time": mtime,
                "download_url": f"/api/v1/multimodal/download/{f}"
            })

        return {
            "status": "success",
            "total_files": len(file_info),
            "files": file_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{filename}")
async def delete_file(filename: str):
    try:
        file_path = f"generated_outputs/{filename}"

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)

        return {
            "status": "success",
            "message": f"Deleted {filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities():
    return {
        "status": "operational",
        "capabilities": {
            "image_generation": {
                "description": "Generate images from text prompts",
                "features": ["custom resolutions", "face preservation", "style control"],
                "endpoint": "/generate_image"
            },
            "video_generation": {
                "description": "Generate videos and GIFs from text",
                "features": ["short videos", "long videos", "GIF export", "face preservation"],
                "endpoint": "/generate_video"
            },
            "audio_generation": {
                "description": "Generate audio, speech, and music",
                "features": ["neural synthesis", "speech synthesis", "music generation"],
                "endpoint": "/generate_audio"
            },
            "face_preservation": {
                "description": "Keep your face unchanged in generated content",
                "features": ["face upload", "automatic blending", "video face tracking"],
                "endpoint": "/upload_face"
            },
            "visual_qa": {
                "description": "Answer questions about images and videos",
                "features": ["image understanding", "video analysis", "content description"],
                "endpoint": "/ask_question"
            },
            "multimodal_understanding": {
                "description": "Understand any type of content",
                "features": ["image captioning", "video description", "audio analysis"],
                "endpoint": "/understand_content"
            }
        }
    }
