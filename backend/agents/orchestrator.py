"""
Agent Orchestrator
Coordinates multiple agents to solve complex tasks
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timezone
import asyncio
from .base_agent import BaseAgent, AgentTask, AgentResult
from .research_agent import ResearchAgent
from .writing_agent import WritingAgent
from loguru import logger


class TaskPlan(BaseModel):
    """Plan for executing a complex task"""
    plan_id: str
    original_query: str
    subtasks: List[AgentTask]
    execution_order: List[int]
    created_at: datetime


class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents to solve complex tasks

    Features:
    - Task decomposition
    - Agent selection and routing
    - Parallel execution
    - Result aggregation
    - Inter-agent communication
    """

    def __init__(self, llm_engine=None, web_search_tool=None):
        self.llm_engine = llm_engine
        self.web_search_tool = web_search_tool
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all available agents"""
        # Research Agent
        research_agent = ResearchAgent(
            llm_engine=self.llm_engine,
            web_search_tool=self.web_search_tool
        )
        self.agents[research_agent.name] = research_agent

        # Writing Agent
        writing_agent = WritingAgent(llm_engine=self.llm_engine)
        self.agents[writing_agent.name] = writing_agent

        logger.info(f"Initialized {len(self.agents)} agents")

    async def execute_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complex query using multiple agents

        Args:
            query: User query
            context: Additional context

        Returns:
            Aggregated results from all agents
        """
        logger.info(f"Orchestrator: Processing query: {query}")

        # Analyze query and create task plan
        plan = await self._create_task_plan(query, context or {})

        # Execute tasks according to plan
        results = await self._execute_plan(plan)

        # Aggregate and synthesize results
        final_response = await self._synthesize_results(query, results)

        return {
            "query": query,
            "response": final_response,
            "agents_used": list(set(r.agent_name for r in results)),
            "tasks_executed": len(results),
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0
        }

    async def _create_task_plan(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[AgentTask]:
        """
        Analyze query and create execution plan

        Uses LLM to decompose complex queries into subtasks
        """
        # For now, use simple rule-based planning
        # In production, use LLM to intelligently decompose tasks
        tasks = []

        # Check if query needs research
        research_keywords = ["research", "find", "search", "what", "who", "when", "where", "how"]
        needs_research = any(keyword in query.lower() for keyword in research_keywords)

        if needs_research:
            tasks.append(AgentTask(
                task_id=str(uuid4()),
                task_type="web_search",
                description=f"Research: {query}",
                input_data={"query": query, "max_results": 5},
                priority=1
            ))

        # Check if query needs writing
        writing_keywords = ["write", "create", "draft", "compose", "generate"]
        needs_writing = any(keyword in query.lower() for keyword in writing_keywords)

        if needs_writing:
            tasks.append(AgentTask(
                task_id=str(uuid4()),
                task_type="generate_content",
                description=f"Write content for: {query}",
                input_data={
                    "topic": query,
                    "style": context.get("style", "professional"),
                    "length": context.get("length", 500)
                },
                priority=2 if needs_research else 1
            ))

        # Default: treat as research question
        if not tasks:
            tasks.append(AgentTask(
                task_id=str(uuid4()),
                task_type="web_search",
                description=f"Answer: {query}",
                input_data={"query": query, "max_results": 3},
                priority=1
            ))

        return tasks

    async def _execute_plan(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """
        Execute task plan

        Handles both sequential and parallel execution
        """
        results = []

        # Group tasks by priority
        priority_groups = {}
        for task in tasks:
            if task.priority not in priority_groups:
                priority_groups[task.priority] = []
            priority_groups[task.priority].append(task)

        # Execute tasks in priority order
        for priority in sorted(priority_groups.keys()):
            group_tasks = priority_groups[priority]

            # Execute tasks in group in parallel
            group_results = await asyncio.gather(*[
                self._route_and_execute(task) for task in group_tasks
            ])

            results.extend(group_results)

        return results

    async def _route_and_execute(self, task: AgentTask) -> AgentResult:
        """
        Route task to appropriate agent and execute

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        # Find suitable agent for task
        suitable_agent = None
        for agent in self.agents.values():
            if agent.can_handle(task.task_type):
                suitable_agent = agent
                break

        if not suitable_agent:
            logger.error(f"No agent found for task type: {task.task_type}")
            return AgentResult(
                task_id=task.task_id,
                agent_name="None",
                success=False,
                output={"error": f"No agent available for task type: {task.task_type}"},
                metadata={},
                execution_time_ms=0
            )

        # Execute task with selected agent
        return await suitable_agent.execute(task)

    async def _synthesize_results(
        self,
        query: str,
        results: List[AgentResult]
    ) -> str:
        """
        Synthesize results from multiple agents into coherent response

        Args:
            query: Original query
            results: Results from all agents

        Returns:
            Synthesized response
        """
        if not results:
            return "I couldn't process your request. Please try again."

        # Collect all successful outputs
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return "I encountered errors while processing your request. Please try again."

        # Build context from all results
        context_parts = []
        for result in successful_results:
            output = result.output
            if isinstance(output, dict):
                # Research results
                if "results" in output:
                    for item in output.get("results", []):
                        context_parts.append(f"- {item.get('title', '')}: {item.get('snippet', '')}")
                # Content generation
                elif "content" in output:
                    context_parts.append(output["content"])
                # Analysis
                elif "analysis" in output:
                    context_parts.append(output["analysis"])
                # Summary
                elif "summary" in output:
                    context_parts.append(output["summary"])

        # If we have LLM, synthesize; otherwise return raw results
        if self.llm_engine and context_parts:
            context = "\n\n".join(context_parts)
            prompt = f"""Based on the following information gathered by multiple AI agents, provide a comprehensive answer to the user's question:

Question: {query}

Information:
{context}

Provide a clear, well-structured response that addresses the question:"""

            synthesis = await self.llm_engine.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000
            )

            return synthesis["content"]
        else:
            # Fallback: return concatenated results
            return "\n\n".join(context_parts) if context_parts else "No results available."

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents

        Returns:
            Agent status information
        """
        return {
            "total_agents": len(self.agents),
            "agents": {
                name: agent.get_info()
                for name, agent in self.agents.items()
            }
        }

    def add_agent(self, agent: BaseAgent):
        """
        Add a new agent to the orchestrator

        Args:
            agent: Agent to add
        """
        self.agents[agent.name] = agent
        logger.info(f"Added agent: {agent.name}")


from pydantic import BaseModel
