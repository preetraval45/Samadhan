"""
RAG Generator
Generates responses using LLM with retrieved context
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class RAGGenerator:
    """
    Response generator with context-aware prompting
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        domain: Optional[str] = None
    ) -> str:
        """
        Build context-aware prompt for LLM

        Args:
            query: User query
            context: Retrieved context chunks
            conversation_history: Previous messages
            domain: Domain filter (healthcare, legal, finance)

        Returns:
            Formatted prompt
        """
        # Build context section
        context_text = "\n\n".join([
            f"[Source: {ctx['metadata']['source']}]\n{ctx['content']}"
            for ctx in context
        ])

        # Build conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])

        # Domain-specific instructions
        domain_instructions = self._get_domain_instructions(domain)

        # Construct full prompt
        prompt = f"""You are Samādhān, an advanced AI decision intelligence assistant.

{domain_instructions}

Use the following context to answer the user's question. If the context doesn't contain relevant information, acknowledge this and provide your best response based on your knowledge.

CONTEXT:
{context_text}

{f"CONVERSATION HISTORY:\n{history_text}\n" if history_text else ""}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a comprehensive, accurate answer
2. Cite sources when using information from context
3. Explain your reasoning and confidence level
4. If uncertain, acknowledge limitations
5. Structure your response clearly

RESPONSE:"""

        return prompt

    def _get_domain_instructions(self, domain: Optional[str]) -> str:
        """Get domain-specific instructions"""
        domain_prompts = {
            "healthcare": "You specialize in healthcare and medical decision support. Provide evidence-based recommendations while noting that this is not a substitute for professional medical advice.",
            "legal": "You specialize in legal analysis and compliance. Provide detailed analysis of legal concepts while noting this is not legal advice.",
            "finance": "You specialize in financial analysis and risk assessment. Provide data-driven insights for informed decision-making.",
            "general": "You provide cross-domain decision intelligence, synthesizing information from multiple sources."
        }

        return domain_prompts.get(domain, domain_prompts["general"])

    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate AI response with context

        Args:
            query: User query
            context: Retrieved context
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum response length

        Returns:
            Response with metadata
        """
        logger.info(f"Generating response with {len(context)} context chunks")

        # Build prompt
        prompt = self.build_prompt(
            query=query,
            context=context,
            conversation_history=kwargs.get('conversation_history'),
            domain=kwargs.get('domain')
        )

        # TODO: Call LLM API
        # response = await self.llm_client.generate(
        #     prompt=prompt,
        #     model=model,
        #     temperature=temperature,
        #     max_tokens=max_tokens
        # )

        # Placeholder response
        response_text = f"Based on the provided context, here's a comprehensive answer to your question: {query}"

        # Extract sources
        sources = [
            {
                "document": ctx["metadata"]["source"],
                "page": ctx["metadata"].get("page", 1),
                "relevance_score": ctx["score"],
                "snippet": ctx["content"][:200] + "..."
            }
            for ctx in context
        ]

        return {
            "response": response_text,
            "sources": sources,
            "model_used": model,
            "tokens_used": 350,  # Placeholder
            "confidence": self._calculate_confidence(context)
        }

    def _calculate_confidence(self, context: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on context quality

        Args:
            context: Retrieved context chunks

        Returns:
            Confidence score (0-1)
        """
        if not context:
            return 0.3  # Low confidence without context

        # Average relevance score
        avg_score = sum(ctx["score"] for ctx in context) / len(context)

        # Adjust based on number of sources
        source_bonus = min(len(context) / 5, 0.2)

        confidence = min(avg_score + source_bonus, 1.0)

        return round(confidence, 2)
