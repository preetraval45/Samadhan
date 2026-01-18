"""
Local/Free LLM Models Provider
Uses HuggingFace Transformers for local model inference
"""

from typing import Dict, Any, Optional, AsyncGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
from loguru import logger


class LocalLLMProvider:
    """
    Provider for running open-source LLMs locally
    Supports models like Phi-3, Mistral, Llama, etc.
    """

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"LocalLLM using device: {self.device}")

        # Default lightweight model for quick responses
        self.default_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def load_model(self, model_name: str = None):
        """Load a model into memory"""
        if model_name is None:
            model_name = self.default_model

        if model_name in self.models:
            return  # Already loaded

        try:
            logger.info(f"Loading model: {model_name}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load model with optimizations for CPU/GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu":
                model = model.to(self.device)

            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            logger.info(f"Model {model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from local model"""
        try:
            if model is None:
                model = self.default_model

            # Load model if not already loaded
            if model not in self.models:
                self.load_model(model)

            tokenizer = self.tokenizers[model]
            llm_model = self.models[model]

            # Format prompt with system message
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"

            # Tokenize input
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode response
            response_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return {
                "content": response_text.strip(),
                "model": model,
                "provider": "local",
                "tokens_used": len(outputs[0]),
                "finish_reason": "stop"
            }

        except Exception as e:
            logger.error(f"Local generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from local model"""
        try:
            if model is None:
                model = self.default_model

            # Load model if not already loaded
            if model not in self.models:
                self.load_model(model)

            tokenizer = self.tokenizers[model]
            llm_model = self.models[model]

            # Format prompt with system message
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"

            # Tokenize input
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Setup streaming
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation parameters
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                **kwargs
            )

            # Run generation in thread
            thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream tokens
            for text in streamer:
                yield {
                    "content": text,
                    "model": model,
                    "provider": "local",
                    "finish_reason": None
                }

            thread.join()

        except Exception as e:
            logger.error(f"Local streaming error: {e}")
            raise

    def get_available_models(self):
        """Get list of recommended free models"""
        return [
            {
                "id": "microsoft/phi-2",
                "name": "Phi-2 (2.7B)",
                "provider": "local",
                "size": "2.7B parameters",
                "description": "Fast, lightweight model by Microsoft"
            },
            {
                "id": "microsoft/Phi-3-mini-4k-instruct",
                "name": "Phi-3 Mini",
                "provider": "local",
                "size": "3.8B parameters",
                "description": "Improved Phi model with instruction tuning"
            },
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                "name": "Mistral 7B Instruct",
                "provider": "local",
                "size": "7B parameters",
                "description": "High-quality open model by Mistral AI"
            },
            {
                "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "name": "TinyLlama Chat",
                "provider": "local",
                "size": "1.1B parameters",
                "description": "Very fast, compact chat model"
            }
        ]
