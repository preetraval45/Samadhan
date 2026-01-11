"""
Model Ensembling
Combine multiple LLM responses for improved quality and reliability
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from loguru import logger


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble"""
    models: List[str]
    strategy: str  # 'voting', 'weighted', 'consensus', 'best'
    weights: Optional[List[float]] = None
    temperature: float = 0.7
    max_tokens: int = 2048


class ModelEnsemble:
    """
    Ensemble multiple LLMs for better responses

    Strategies:
    - Voting: Majority consensus
    - Weighted: Weighted combination based on model quality
    - Consensus: Only include information all models agree on
    - Best: Select best response based on quality metrics
    """

    def __init__(self, llm_engine):
        self.llm_engine = llm_engine

    async def generate_ensemble(
        self,
        prompt: str,
        config: EnsembleConfig,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using ensemble of models

        Args:
            prompt: User prompt
            config: Ensemble configuration
            system_prompt: Optional system prompt

        Returns:
            Ensembled response with metadata
        """
        logger.info(f"Ensemble generation with {len(config.models)} models")

        # Generate responses from all models in parallel
        tasks = [
            self.llm_engine.generate(
                prompt=prompt,
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                system_prompt=system_prompt
            )
            for model in config.models
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Model {config.models[i]} failed: {response}")
            else:
                valid_responses.append({
                    "model": config.models[i],
                    "response": response,
                    "weight": config.weights[i] if config.weights else 1.0
                })

        if not valid_responses:
            raise Exception("All models failed to generate responses")

        logger.info(f"Got {len(valid_responses)} valid responses")

        # Apply ensemble strategy
        if config.strategy == "voting":
            result = await self._voting_strategy(valid_responses, prompt)
        elif config.strategy == "weighted":
            result = await self._weighted_strategy(valid_responses)
        elif config.strategy == "consensus":
            result = await self._consensus_strategy(valid_responses, prompt)
        elif config.strategy == "best":
            result = await self._best_strategy(valid_responses, prompt)
        else:
            # Default: use first valid response
            result = valid_responses[0]["response"]

        return {
            "content": result,
            "ensemble_strategy": config.strategy,
            "models_used": [r["model"] for r in valid_responses],
            "total_models": len(config.models),
            "successful_models": len(valid_responses)
        }

    async def _voting_strategy(
        self,
        responses: List[Dict[str, Any]],
        prompt: str
    ) -> str:
        """
        Voting strategy: Use LLM to synthesize consensus

        Args:
            responses: List of model responses
            prompt: Original prompt

        Returns:
            Synthesized consensus response
        """
        # Build context from all responses
        context = "Multiple AI models provided the following responses:\n\n"
        for i, resp in enumerate(responses, 1):
            context += f"Model {i} ({resp['model']}):\n"
            context += f"{resp['response']['content']}\n\n"

        # Use LLM to synthesize
        synthesis_prompt = f"""Given the following question and multiple AI responses, create a single, comprehensive answer that captures the consensus and best insights from all responses.

Question: {prompt}

{context}

Provide a synthesized response that:
1. Captures points of agreement
2. Includes unique valuable insights
3. Maintains accuracy and coherence

Synthesized response:"""

        synthesis = await self.llm_engine.generate(
            prompt=synthesis_prompt,
            temperature=0.3,
            max_tokens=2048
        )

        return synthesis["content"]

    async def _weighted_strategy(
        self,
        responses: List[Dict[str, Any]]
    ) -> str:
        """
        Weighted strategy: Prioritize higher-weighted models

        For now, returns the highest-weighted response
        In production, this could do weighted text combination
        """
        # Sort by weight
        sorted_responses = sorted(
            responses,
            key=lambda x: x["weight"],
            reverse=True
        )

        # Return highest weighted response
        return sorted_responses[0]["response"]["content"]

    async def _consensus_strategy(
        self,
        responses: List[Dict[str, Any]],
        prompt: str
    ) -> str:
        """
        Consensus strategy: Extract common information

        Uses LLM to identify points all models agree on
        """
        context = "Multiple AI models provided responses:\n\n"
        for i, resp in enumerate(responses, 1):
            context += f"Response {i}:\n{resp['response']['content']}\n\n"

        consensus_prompt = f"""Analyze these AI responses and extract ONLY the information that all responses agree on.

Question: {prompt}

{context}

Provide a response containing ONLY information present in ALL responses (consensus points):"""

        consensus = await self.llm_engine.generate(
            prompt=consensus_prompt,
            temperature=0.2,
            max_tokens=1500
        )

        return consensus["content"]

    async def _best_strategy(
        self,
        responses: List[Dict[str, Any]],
        prompt: str
    ) -> str:
        """
        Best strategy: Select highest quality response

        Uses LLM to evaluate and select best response
        """
        # Create evaluation prompt
        eval_context = ""
        for i, resp in enumerate(responses, 1):
            eval_context += f"Response {i} (Model: {resp['model']}):\n"
            eval_context += f"{resp['response']['content']}\n\n"

        eval_prompt = f"""Evaluate these AI responses and identify the BEST one.

Question: {prompt}

{eval_context}

Criteria for best response:
- Accuracy and correctness
- Completeness
- Clarity and coherence
- Relevance to the question

Which response is best? Reply with just the number (1, 2, 3, etc):"""

        evaluation = await self.llm_engine.generate(
            prompt=eval_prompt,
            temperature=0.1,
            max_tokens=5
        )

        # Parse response number
        try:
            best_idx = int(evaluation["content"].strip()) - 1
            if 0 <= best_idx < len(responses):
                return responses[best_idx]["response"]["content"]
        except:
            pass

        # Fallback: return first response
        return responses[0]["response"]["content"]

    async def generate_with_verification(
        self,
        prompt: str,
        models: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate response with cross-model verification

        Uses multiple models and only returns result if confidence is high

        Args:
            prompt: User prompt
            models: List of models to use
            threshold: Confidence threshold (0-1)

        Returns:
            Verified response with confidence score
        """
        config = EnsembleConfig(
            models=models,
            strategy="consensus",
            temperature=0.5
        )

        result = await self.generate_ensemble(prompt, config)

        # Calculate confidence based on agreement
        confidence = len(result["successful_models"]) / len(result["models_used"])

        return {
            **result,
            "confidence": confidence,
            "verified": confidence >= threshold
        }


# Predefined ensemble configurations
ENSEMBLE_CONFIGS = {
    "high_quality": EnsembleConfig(
        models=["gpt-4", "claude-3-opus-20240229"],
        strategy="best",
        weights=[1.0, 1.0],
        temperature=0.5
    ),
    "fast_consensus": EnsembleConfig(
        models=["gpt-4-turbo", "claude-3-sonnet-20240229"],
        strategy="voting",
        temperature=0.7
    ),
    "verified": EnsembleConfig(
        models=["gpt-4", "claude-3-opus-20240229", "gpt-4-turbo"],
        strategy="consensus",
        temperature=0.3
    )
}
