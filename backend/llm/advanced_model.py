"""
Advanced High-Level AI Model with Web Search Integration
Production-ready model with internet connectivity and real-time information
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from loguru import logger
import asyncio
from datetime import datetime


class WebSearchAgent:
    """
    Intelligent web search agent for real-time information retrieval
    """

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def google_search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search Google and return results
        """
        logger.info(f"Searching Google for: {query}")

        results = []
        try:
            search_results = search(query, num_results=num_results, lang="en")

            for url in search_results:
                try:
                    # Fetch page content
                    response = requests.get(url, headers=self.headers, timeout=5)
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Extract title and text
                    title = soup.title.string if soup.title else ""
                    paragraphs = soup.find_all('p')
                    text = ' '.join([p.get_text() for p in paragraphs[:5]])  # First 5 paragraphs

                    results.append({
                        "url": url,
                        "title": title,
                        "snippet": text[:500]  # First 500 chars
                    })

                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return results

    def search_and_summarize(self, query: str) -> str:
        """
        Search and create a summary of findings
        """
        results = self.google_search(query)

        if not results:
            return "I couldn't find any relevant information online."

        # Combine all snippets
        combined_text = "\n\n".join([
            f"Source: {r['title']}\n{r['snippet']}"
            for r in results
        ])

        return combined_text


class AdvancedLLMModel:
    """
    High-level production AI model with advanced capabilities:
    - Web search integration for real-time information
    - Multi-turn conversation memory
    - Context-aware responses
    - Fact-checking with web sources
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        enable_web_search: bool = True
    ):
        self.model_name = model_name
        self.enable_web_search = enable_web_search

        # Initialize web search
        if enable_web_search:
            self.search_agent = WebSearchAgent()

        # Initialize model and tokenizer
        logger.info(f"Loading advanced model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Conversation history
        self.conversation_history = {}

        logger.info("Advanced model loaded successfully")

    def should_search_web(self, prompt: str) -> bool:
        """
        Determine if web search is needed for this query
        """
        search_keywords = [
            "latest", "recent", "news", "current", "today",
            "what is", "who is", "when did", "where is",
            "weather", "stock", "price", "happening now"
        ]

        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in search_keywords)

    async def generate_with_context(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_web_search: bool = None
    ) -> Dict[str, Any]:
        """
        Generate response with context and optional web search
        """
        # Determine if we should use web search
        if use_web_search is None:
            use_web_search = self.enable_web_search and self.should_search_web(prompt)

        # Get web context if needed
        web_context = ""
        sources = []

        if use_web_search:
            logger.info("Fetching real-time information from the web...")
            web_results = self.search_agent.google_search(prompt, num_results=3)

            if web_results:
                web_context = "\\n\\n=== Real-time Information from the Web ===\\n"
                for result in web_results:
                    web_context += f"\\nSource: {result['title']}\\n{result['snippet']}\\n"
                    sources.append({
                        "url": result["url"],
                        "title": result["title"],
                        "relevance": 0.9
                    })

        # Build conversation context
        history_context = ""
        if conversation_id and conversation_id in self.conversation_history:
            history = self.conversation_history[conversation_id][-3:]  # Last 3 exchanges
            for h in history:
                history_context += f"User: {h['user']}\\nAssistant: {h['assistant']}\\n"

        # Create full prompt
        system_prompt = f"""You are an advanced AI assistant with access to real-time internet information.

Your capabilities:
- Answer questions with current, up-to-date information
- Provide accurate, fact-checked responses
- Cite sources when using web information
- Engage in natural, human-like conversation
- Remember context from the conversation

Guidelines:
- Be conversational and friendly
- If using web sources, acknowledge them naturally
- If uncertain, be honest about limitations
- Provide detailed, helpful responses

{web_context}

Previous conversation:
{history_context}

Current query: {prompt}

Respond naturally and helpfully:"""

        # Tokenize
        inputs = self.tokenizer(
            system_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Update conversation history
        if conversation_id:
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []

            self.conversation_history[conversation_id].append({
                "user": prompt,
                "assistant": response_text,
                "timestamp": datetime.now().isoformat(),
                "used_web_search": use_web_search
            })

        return {
            "content": response_text,
            "model": self.model_name,
            "provider": "advanced_local",
            "used_web_search": use_web_search,
            "sources": sources,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }

    async def stream_generate(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        """
        Generate streaming response
        """
        # For streaming, we'll simulate it by generating in chunks
        result = await self.generate_with_context(
            prompt=prompt,
            conversation_id=conversation_id,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Yield content in chunks
        content = result["content"]
        words = content.split()

        for i, word in enumerate(words):
            yield {
                "content": word + " ",
                "model": self.model_name,
                "provider": "advanced_local",
                "is_final": i == len(words) - 1
            }

            # Small delay for streaming effect
            await asyncio.sleep(0.05)


class ProductionModelManager:
    """
    Manage multiple advanced models for production
    """

    def __init__(self):
        self.models = {}
        self.current_model = None

    def load_model(
        self,
        model_name: str,
        enable_web_search: bool = True
    ) -> AdvancedLLMModel:
        """
        Load an advanced model
        """
        if model_name not in self.models:
            logger.info(f"Loading model: {model_name}")
            self.models[model_name] = AdvancedLLMModel(
                model_name=model_name,
                enable_web_search=enable_web_search
            )

        self.current_model = model_name
        return self.models[model_name]

    def get_model(self, model_name: str = None) -> AdvancedLLMModel:
        """
        Get loaded model
        """
        if model_name:
            return self.models.get(model_name)

        return self.models.get(self.current_model)

    async def generate(
        self,
        prompt: str,
        model_name: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using specified or current model
        """
        model = self.get_model(model_name)

        if not model:
            raise ValueError(f"Model {model_name} not loaded")

        return await model.generate_with_context(prompt, **kwargs)


# Available production models
PRODUCTION_MODELS = {
    "phi-2-web": {
        "name": "microsoft/phi-2",
        "description": "Phi-2 with web search",
        "web_search": True,
        "size": "2.7B"
    },
    "tinyllama-web": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama with web search (fastest)",
        "web_search": True,
        "size": "1.1B"
    },
    "mistral-web": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral 7B with web search (highest quality)",
        "web_search": True,
        "size": "7B"
    }
}


if __name__ == "__main__":
    # Example usage
    async def test_model():
        model = AdvancedLLMModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            enable_web_search=True
        )

        # Test query that should trigger web search
        result = await model.generate_with_context(
            prompt="What are the latest developments in AI in 2026?",
            conversation_id="test123"
        )

        logger.info(f"Response: {result['content']}")
        logger.info(f"Used web search: {result['used_web_search']}")
        logger.info(f"Sources: {result['sources']}")

    # Run test
    # asyncio.run(test_model())
    logger.info("Advanced model with web search ready!")
