from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
import os
import json

from custom_ai import (
    CustomTokenizer,
    CustomTransformer,
    CustomTrainer,
    DataPreprocessor,
    InferenceEngine,
    ConversationManager
)


router = APIRouter()


model_cache = {
    'tokenizer': None,
    'model': None,
    'inference_engine': None,
    'conversation_manager': None
}


class TrainingRequest(BaseModel):
    dataset_path: str
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    vocab_size: int = 30000


class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    max_length: int = 150


class ModelInitRequest(BaseModel):
    vocab_size: int = 30000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048



@router.post("/init_model")
async def initialize_model(request: ModelInitRequest):
    try:
        global model_cache

        tokenizer = CustomTokenizer(vocab_size=request.vocab_size)

        model = CustomTransformer(
            vocab_size=request.vocab_size,
            d_model=request.d_model,
            num_heads=request.num_heads,
            num_layers=request.num_layers,
            d_ff=request.d_ff
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        inference_engine = InferenceEngine(model, tokenizer, device=device)
        conversation_manager = ConversationManager(inference_engine)

        model_cache['tokenizer'] = tokenizer
        model_cache['model'] = model
        model_cache['inference_engine'] = inference_engine
        model_cache['conversation_manager'] = conversation_manager

        return {
            "status": "success",
            "message": "Model initialized successfully",
            "device": device,
            "vocab_size": request.vocab_size,
            "d_model": request.d_model,
            "num_layers": request.num_layers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    try:
        if model_cache['tokenizer'] is None or model_cache['model'] is None:
            raise HTTPException(status_code=400, detail="Model not initialized")

        preprocessor = DataPreprocessor()

        if request.dataset_path.endswith('.json'):
            texts = preprocessor.load_json_dataset(request.dataset_path)
        else:
            texts = preprocessor.load_text_files([request.dataset_path])

        model_cache['tokenizer'].train(texts)

        train_texts, val_texts, test_texts = preprocessor.split_dataset(texts)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = CustomTrainer(
            model_cache['model'],
            model_cache['tokenizer'],
            device=device,
            learning_rate=request.learning_rate
        )

        def train_background():
            trainer.train(
                train_texts,
                val_texts,
                epochs=request.epochs,
                batch_size=request.batch_size,
                save_dir='checkpoints'
            )

        background_tasks.add_task(train_background)

        return {
            "status": "training_started",
            "message": f"Training started with {len(train_texts)} samples",
            "epochs": request.epochs,
            "batch_size": request.batch_size
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        if model_cache['inference_engine'] is None:
            raise HTTPException(status_code=400, detail="Model not initialized")

        generated_text = model_cache['inference_engine'].generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

        return {
            "status": "success",
            "prompt": request.prompt,
            "generated_text": generated_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        if model_cache['conversation_manager'] is None:
            raise HTTPException(status_code=400, detail="Model not initialized")

        response = model_cache['conversation_manager'].get_response(
            request.conversation_id,
            request.message
        )

        history = model_cache['conversation_manager'].get_history(request.conversation_id)

        return {
            "status": "success",
            "response": response,
            "conversation_id": request.conversation_id,
            "history_length": len(history)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/load_checkpoint")
async def load_checkpoint(checkpoint_path: str):
    try:
        if model_cache['model'] is None or model_cache['tokenizer'] is None:
            raise HTTPException(status_code=400, detail="Model not initialized")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_cache['model'].load_state_dict(checkpoint['model_state_dict'])

        tokenizer_path = checkpoint_path.replace('.pt', '_tokenizer.pkl')
        if os.path.exists(tokenizer_path):
            model_cache['tokenizer'].load(tokenizer_path)

        model_cache['inference_engine'] = InferenceEngine(
            model_cache['model'],
            model_cache['tokenizer'],
            device=device
        )
        model_cache['conversation_manager'] = ConversationManager(
            model_cache['inference_engine']
        )

        return {
            "status": "success",
            "message": f"Loaded checkpoint from {checkpoint_path}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/save_checkpoint")
async def save_checkpoint(checkpoint_path: str):
    try:
        if model_cache['model'] is None or model_cache['tokenizer'] is None:
            raise HTTPException(status_code=400, detail="Model not initialized")

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            'model_state_dict': model_cache['model'].state_dict(),
            'model_config': {
                'vocab_size': model_cache['model'].vocab_size,
                'd_model': model_cache['model'].d_model,
            }
        }

        torch.save(checkpoint, checkpoint_path)

        tokenizer_path = checkpoint_path.replace('.pt', '_tokenizer.pkl')
        model_cache['tokenizer'].save(tokenizer_path)

        return {
            "status": "success",
            "message": f"Saved checkpoint to {checkpoint_path}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/model_info")
async def get_model_info():
    if model_cache['model'] is None:
        return {
            "status": "not_initialized",
            "message": "Model not initialized"
        }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    param_count = sum(p.numel() for p in model_cache['model'].parameters())

    return {
        "status": "initialized",
        "vocab_size": model_cache['model'].vocab_size,
        "d_model": model_cache['model'].d_model,
        "parameter_count": param_count,
        "device": device
    }



@router.post("/clear_conversation")
async def clear_conversation(conversation_id: str):
    try:
        if model_cache['conversation_manager'] is None:
            raise HTTPException(status_code=400, detail="Model not initialized")

        model_cache['conversation_manager'].clear_conversation(conversation_id)

        return {
            "status": "success",
            "message": f"Cleared conversation {conversation_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
