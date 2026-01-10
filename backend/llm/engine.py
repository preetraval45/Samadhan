"""
LLM Engine
Multi-provider LLM client with unified interface
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import openai
from anthropic import Anthropic
from loguru import logger
from core.config import settings


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class LLMEngine:
    """
    Unified LLM interface supporting multiple providers
    """

    def __init__(self):
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available LLM providers"""
        if settings.OPENAI_API_KEY:
            self.providers[LLMProvider.OPENAI] = openai.OpenAI(
                api_key=settings.OPENAI_API_KEY
            )
            logger.info("OpenAI provider initialized")

        if settings.ANTHROPIC_API_KEY:
            self.providers[LLMProvider.ANTHROPIC] = Anthropic(
                api_key=settings.ANTHROPIC_API_KEY
            )
            logger.info("Anthropic provider initialized")

    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response from LLM

        Args:
            prompt: User prompt
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response length
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Response with metadata
        """
        provider = self._get_provider_for_model(model)

        if provider == LLMProvider.OPENAI:
            return await self._generate_openai(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            )
        elif provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

    def _get_provider_for_model(self, model: str) -> LLMProvider:
        """Determine provider from model name"""
        if model.startswith("gpt-"):
            return LLMProvider.OPENAI
        elif model.startswith("claude-"):
            return LLMProvider.ANTHROPIC
        else:
            return LLMProvider.OPENAI  # Default

    async def _generate_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        try:
            client = self.providers[LLMProvider.OPENAI]

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return {
                "content": response.choices[0].message.content,
                "model": model,
                "provider": "openai",
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def _generate_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Anthropic Claude"""
        try:
            client = self.providers[LLMProvider.ANTHROPIC]

            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return {
                "content": message.content[0].text,
                "model": model,
                "provider": "anthropic",
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "finish_reason": message.stop_reason
            }

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise

    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """
        Generate embeddings for texts

        Args:
            texts: List of texts to embed
            model: Embedding model

        Returns:
            List of embedding vectors
        """
        try:
            client = self.providers[LLMProvider.OPENAI]

            response = client.embeddings.create(
                model=model,
                input=texts
            )

            embeddings = [data.embedding for data in response.data]
            logger.info(f"Generated embeddings for {len(texts)} texts")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []

        if LLMProvider.OPENAI in self.providers:
            models.extend([
                {
                    "id": "gpt-4",
                    "name": "GPT-4",
                    "provider": "openai",
                    "context_length": 8192,
                    "capabilities": ["text", "code"]
                },
                {
                    "id": "gpt-4-turbo",
                    "name": "GPT-4 Turbo",
                    "provider": "openai",
                    "context_length": 128000,
                    "capabilities": ["text", "code", "vision"]
                }
            ])

        if LLMProvider.ANTHROPIC in self.providers:
            models.extend([
                {
                    "id": "claude-3-opus-20240229",
                    "name": "Claude 3 Opus",
                    "provider": "anthropic",
                    "context_length": 200000,
                    "capabilities": ["text", "code", "vision"]
                },
                {
                    "id": "claude-3-sonnet-20240229",
                    "name": "Claude 3 Sonnet",
                    "provider": "anthropic",
                    "context_length": 200000,
                    "capabilities": ["text", "code", "vision"]
                }
            ])

        return models
