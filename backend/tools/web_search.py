"""
Web Search Tool
Enables AI to search the web for real-time information
"""

from typing import List, Dict, Any, Optional
import aiohttp
from datetime import datetime, timezone
from loguru import logger
from core.config import settings


class WebSearchTool:
    """
    Web search integration for AI-powered information retrieval

    Supports multiple search providers:
    - DuckDuckGo (no API key required)
    - Google Custom Search API
    - Bing Search API
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def search_duckduckgo(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo (no API key required)

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, snippet, url
        """
        try:
            session = await self._get_session()

            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"DuckDuckGo API returned status {response.status}")

                data = await response.json()
                results = []

                # Extract abstract
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("Heading", "DuckDuckGo Abstract"),
                        "snippet": data.get("Abstract"),
                        "url": data.get("AbstractURL", ""),
                        "source": "DuckDuckGo"
                    })

                # Extract related topics
                for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({
                            "title": topic.get("Text", "").split(" - ")[0],
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                            "source": "DuckDuckGo"
                        })

                logger.info(f"DuckDuckGo search completed: {len(results)} results for '{query}'")
                return results[:max_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def search_google(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using Google Custom Search API

        Requires: GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in settings

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        if not hasattr(settings, 'GOOGLE_API_KEY') or not hasattr(settings, 'GOOGLE_SEARCH_ENGINE_ID'):
            logger.warning("Google Search API credentials not configured")
            return []

        try:
            session = await self._get_session()

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": settings.GOOGLE_API_KEY,
                "cx": settings.GOOGLE_SEARCH_ENGINE_ID,
                "q": query,
                "num": min(max_results, 10)
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Google API returned status {response.status}")

                data = await response.json()
                results = []

                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": item.get("link", ""),
                        "source": "Google"
                    })

                logger.info(f"Google search completed: {len(results)} results for '{query}'")
                return results

        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []

    async def search(
        self,
        query: str,
        max_results: int = 5,
        provider: str = "duckduckgo"
    ) -> Dict[str, Any]:
        """
        Universal search method

        Args:
            query: Search query
            max_results: Maximum number of results
            provider: Search provider (duckduckgo, google)

        Returns:
            Search results with metadata
        """
        logger.info(f"Web search: '{query}' using {provider}")

        if provider == "google":
            results = await self.search_google(query, max_results)
        else:
            results = await self.search_duckduckgo(query, max_results)

        return {
            "query": query,
            "provider": provider,
            "results": results,
            "total_results": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def search_and_summarize(
        self,
        query: str,
        llm_engine,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search the web and use LLM to summarize findings

        Args:
            query: Search query
            llm_engine: LLM engine instance for summarization
            max_results: Maximum search results

        Returns:
            Search results with AI-generated summary
        """
        # Perform web search
        search_data = await self.search(query, max_results)

        if not search_data["results"]:
            return {
                **search_data,
                "summary": "No web search results found for this query.",
                "confidence": 0.0
            }

        # Build context from search results
        context = f"Web search results for: {query}\n\n"
        for i, result in enumerate(search_data["results"], 1):
            context += f"{i}. {result['title']}\n"
            context += f"   {result['snippet']}\n"
            context += f"   Source: {result['url']}\n\n"

        # Generate summary using LLM
        try:
            summary_prompt = f"""Based on the following web search results, provide a comprehensive and accurate summary.
Include key facts, cite sources, and note if information is conflicting or uncertain.

{context}

Provide a clear, factual summary:"""

            llm_response = await llm_engine.generate(
                prompt=summary_prompt,
                temperature=0.3,
                max_tokens=500
            )

            return {
                **search_data,
                "summary": llm_response["content"],
                "confidence": 0.85,
                "model_used": llm_response["model"]
            }

        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return {
                **search_data,
                "summary": "Search results retrieved but summary generation failed.",
                "confidence": 0.5
            }


# Tool description for AI function calling
WEB_SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for real-time information on any topic. Use this when you need current data, recent events, or information not in your training data.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default: 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
