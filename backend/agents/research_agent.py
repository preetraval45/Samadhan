"""
Research Agent
Specialized in gathering and synthesizing information from multiple sources
"""

from typing import List, Dict, Any
from datetime import datetime, timezone
import time
from .base_agent import BaseAgent, AgentTask, AgentResult, AgentCapability
from loguru import logger


class ResearchAgent(BaseAgent):
    """
    Agent specialized in research tasks

    Capabilities:
    - Web search and information gathering
    - Document analysis
    - Fact checking
    - Source citation
    """

    def __init__(self, llm_engine=None, web_search_tool=None):
        super().__init__(
            name="ResearchAgent",
            role="Information Researcher",
            description="Gathers and synthesizes information from multiple sources"
        )
        self.llm_engine = llm_engine
        self.web_search_tool = web_search_tool

    def get_capabilities(self) -> List[AgentCapability]:
        """Define research agent capabilities"""
        return [
            AgentCapability(
                name="web_search",
                description="Search the web for current information",
                parameters={"query": "string", "max_results": "int"}
            ),
            AgentCapability(
                name="fact_check",
                description="Verify claims and check facts",
                parameters={"claim": "string", "sources": "list"}
            ),
            AgentCapability(
                name="summarize_sources",
                description="Summarize information from multiple sources",
                parameters={"sources": "list", "focus": "string"}
            ),
            AgentCapability(
                name="comparative_analysis",
                description="Compare and contrast information from different sources",
                parameters={"topic": "string", "sources": "list"}
            )
        ]

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute research task

        Args:
            task: Research task to execute

        Returns:
            Research results
        """
        start_time = time.time()
        logger.info(f"[{self.name}] Starting task: {task.task_type}")

        try:
            if task.task_type == "web_search":
                output = await self._perform_web_search(task.input_data)
            elif task.task_type == "fact_check":
                output = await self._fact_check(task.input_data)
            elif task.task_type == "summarize_sources":
                output = await self._summarize_sources(task.input_data)
            elif task.task_type == "comparative_analysis":
                output = await self._comparative_analysis(task.input_data)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            execution_time = (time.time() - start_time) * 1000

            result = AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=True,
                output=output,
                metadata={
                    "task_type": task.task_type,
                    "sources_consulted": output.get("sources_count", 0)
                },
                execution_time_ms=execution_time
            )

            self.log_result(result)
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Task failed: {str(e)}")
            execution_time = (time.time() - start_time) * 1000

            result = AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                output={"error": str(e)},
                metadata={"task_type": task.task_type},
                execution_time_ms=execution_time
            )

            self.log_result(result)
            return result

    async def _perform_web_search(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search"""
        query = input_data.get("query", "")
        max_results = input_data.get("max_results", 5)

        if not self.web_search_tool:
            return {
                "error": "Web search tool not available",
                "results": [],
                "sources_count": 0
            }

        search_results = await self.web_search_tool.search(query, max_results)

        return {
            "query": query,
            "results": search_results.get("results", []),
            "sources_count": len(search_results.get("results", [])),
            "provider": search_results.get("provider", "unknown")
        }

    async def _fact_check(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify claims using web search and LLM analysis"""
        claim = input_data.get("claim", "")

        if not self.web_search_tool or not self.llm_engine:
            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "explanation": "Fact checking tools not available"
            }

        # Search for information about the claim
        search_results = await self.web_search_tool.search(
            query=f"fact check: {claim}",
            max_results=5
        )

        # Build context from search results
        context = "\n\n".join([
            f"Source: {result['title']}\n{result['snippet']}"
            for result in search_results.get("results", [])
        ])

        # Use LLM to analyze the claim
        prompt = f"""Fact check the following claim using the provided sources:

Claim: {claim}

Sources:
{context}

Provide:
1. Verdict (True/False/Partially True/Unverified)
2. Confidence level (0-1)
3. Explanation with source citations"""

        analysis = await self.llm_engine.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=500
        )

        return {
            "claim": claim,
            "analysis": analysis["content"],
            "sources": search_results.get("results", []),
            "sources_count": len(search_results.get("results", []))
        }

    async def _summarize_sources(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize information from multiple sources"""
        sources = input_data.get("sources", [])
        focus = input_data.get("focus", "")

        if not self.llm_engine:
            return {
                "summary": "Summary generation not available",
                "sources_count": 0
            }

        # Build context from sources
        context = "\n\n".join([
            f"Source {i+1}: {source.get('title', 'Unknown')}\n{source.get('snippet', source.get('content', ''))}"
            for i, source in enumerate(sources)
        ])

        prompt = f"""Summarize the following sources{' focusing on: ' + focus if focus else ''}:

{context}

Provide a comprehensive summary with key points and citations."""

        summary = await self.llm_engine.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=800
        )

        return {
            "summary": summary["content"],
            "sources_count": len(sources),
            "focus": focus
        }

    async def _comparative_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare information from different sources"""
        topic = input_data.get("topic", "")
        sources = input_data.get("sources", [])

        if not self.llm_engine:
            return {
                "analysis": "Comparative analysis not available",
                "sources_count": 0
            }

        # Build context
        context = "\n\n".join([
            f"Source {i+1}: {source.get('title', 'Unknown')}\n{source.get('snippet', source.get('content', ''))}"
            for i, source in enumerate(sources)
        ])

        prompt = f"""Compare and contrast the following sources on the topic: {topic}

{context}

Provide:
1. Key agreements across sources
2. Key disagreements or contradictions
3. Unique perspectives from each source
4. Overall synthesis"""

        analysis = await self.llm_engine.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1000
        )

        return {
            "topic": topic,
            "analysis": analysis["content"],
            "sources_count": len(sources)
        }
