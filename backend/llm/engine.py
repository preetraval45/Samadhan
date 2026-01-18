"""
LLM Engine
Multi-provider LLM client with unified interface
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
from enum import Enum
import openai
from anthropic import Anthropic
from loguru import logger
from core.config import settings
from llm.local_models import LocalLLMProvider
from llm.advanced_model import AdvancedLLMModel, ProductionModelManager


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    ADVANCED = "advanced"  # Advanced model with web search (default)
    LOCAL = "local"  # Free, open-source models
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class LLMEngine:
    """
    Unified LLM interface supporting multiple providers
    Prioritizes free, local models
    """

    def __init__(self):
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Initialize advanced model with web search (highest priority)
        try:
            self.providers[LLMProvider.ADVANCED] = AdvancedLLMModel(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                enable_web_search=True
            )
            logger.info("Advanced LLM provider initialized with web search")
        except Exception as e:
            logger.warning(f"Could not initialize advanced provider: {e}")

        # Always initialize local/free models as fallback
        try:
            self.providers[LLMProvider.LOCAL] = LocalLLMProvider()
            logger.info("Local LLM provider initialized (free models)")
        except Exception as e:
            logger.warning(f"Could not initialize local provider: {e}")

        # Optional: Initialize paid API providers if keys are configured
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here":
            self.providers[LLMProvider.OPENAI] = openai.OpenAI(
                api_key=settings.OPENAI_API_KEY
            )
            logger.info("OpenAI provider initialized")

        if settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
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

        if provider == LLMProvider.ADVANCED:
            return await self._generate_advanced(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            )
        elif provider == LLMProvider.LOCAL:
            return await self._generate_local(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            )
        elif provider == LLMProvider.OPENAI:
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
            return LLMProvider.OPENAI if LLMProvider.OPENAI in self.providers else LLMProvider.ADVANCED
        elif model.startswith("claude-"):
            return LLMProvider.ANTHROPIC if LLMProvider.ANTHROPIC in self.providers else LLMProvider.ADVANCED
        else:
            # Default to advanced model with web search
            return LLMProvider.ADVANCED if LLMProvider.ADVANCED in self.providers else LLMProvider.LOCAL

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

    async def _generate_advanced(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using advanced model with web search"""
        try:
            client = self.providers[LLMProvider.ADVANCED]

            # Build full prompt with system message
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"

            result = await client.generate_with_context(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                conversation_id=kwargs.get("conversation_id"),
                use_web_search=kwargs.get("use_web_search")
            )

            # Convert to standard format
            return {
                "content": result["content"],
                "model": result["model"],
                "provider": "advanced",
                "tokens_used": max_tokens,  # Estimate
                "finish_reason": "stop",
                "sources": result.get("sources", []),
                "used_web_search": result.get("used_web_search", False)
            }
        except Exception as e:
            logger.error(f"Advanced generation error: {e}")
            raise

    async def _generate_local(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using local models"""
        try:
            client = self.providers[LLMProvider.LOCAL]
            return await client.generate(
                prompt=prompt,
                model=model if not model.startswith("gpt-") and not model.startswith("claude-") else None,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Local generation error: {e}")
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

    async def generate_stream(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming response from LLM

        Args:
            prompt: User prompt
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response length
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Yields:
            Token chunks with metadata
        """
        provider = self._get_provider_for_model(model)

        if provider == LLMProvider.ADVANCED:
            async for chunk in self._generate_advanced_stream(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            ):
                yield chunk
        elif provider == LLMProvider.LOCAL:
            async for chunk in self._generate_local_stream(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            ):
                yield chunk
        elif provider == LLMProvider.OPENAI:
            async for chunk in self._generate_openai_stream(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            ):
                yield chunk
        elif provider == LLMProvider.ANTHROPIC:
            async for chunk in self._generate_anthropic_stream(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            ):
                yield chunk
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def _generate_openai_stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response using OpenAI"""
        try:
            client = self.providers[LLMProvider.OPENAI]

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "model": model,
                        "provider": "openai",
                        "finish_reason": chunk.choices[0].finish_reason
                    }

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def _generate_local_stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response using local models"""
        try:
            client = self.providers[LLMProvider.LOCAL]
            async for chunk in client.generate_stream(
                prompt=prompt,
                model=model if not model.startswith("gpt-") and not model.startswith("claude-") else None,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                **kwargs
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Local streaming error: {e}")
            raise

    async def _generate_anthropic_stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response using Anthropic Claude"""
        try:
            client = self.providers[LLMProvider.ANTHROPIC]

            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield {
                        "content": text,
                        "model": model,
                        "provider": "anthropic",
                        "finish_reason": None
                    }

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
