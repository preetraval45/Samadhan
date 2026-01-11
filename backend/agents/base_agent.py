"""
Base Agent Class
Foundation for all specialized agents in the multi-agent system
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from loguru import logger


class AgentCapability(BaseModel):
    """Agent capability definition"""
    name: str
    description: str
    parameters: Dict[str, Any] = {}


class AgentTask(BaseModel):
    """Task for an agent to execute"""
    task_id: str
    task_type: str
    description: str
    input_data: Dict[str, Any]
    priority: int = Field(1, ge=1, le=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResult(BaseModel):
    """Result from agent execution"""
    task_id: str
    agent_name: str
    success: bool
    output: Any
    metadata: Dict[str, Any] = {}
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float


class BaseAgent(ABC):
    """
    Base class for all specialized agents

    Each agent has:
    - Unique name and role
    - Specific capabilities
    - Ability to execute tasks
    - Communication with other agents
    """

    def __init__(self, name: str, role: str, description: str):
        self.name = name
        self.role = role
        self.description = description
        self.capabilities: List[AgentCapability] = []
        self.task_history: List[AgentResult] = []

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute a task

        Args:
            task: Task to execute

        Returns:
            Result of task execution
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Get list of agent capabilities

        Returns:
            List of capabilities this agent can perform
        """
        pass

    def can_handle(self, task_type: str) -> bool:
        """
        Check if agent can handle a specific task type

        Args:
            task_type: Type of task

        Returns:
            True if agent can handle the task
        """
        capabilities = self.get_capabilities()
        return any(cap.name == task_type for cap in capabilities)

    def log_result(self, result: AgentResult):
        """
        Log task result to history

        Args:
            result: Task result to log
        """
        self.task_history.append(result)
        logger.info(f"[{self.name}] Task {result.task_id} completed: {result.success}")

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information

        Returns:
            Agent metadata
        """
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "capabilities": [cap.dict() for cap in self.get_capabilities()],
            "tasks_completed": len(self.task_history),
            "success_rate": self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate success rate from task history"""
        if not self.task_history:
            return 0.0
        successful = sum(1 for result in self.task_history if result.success)
        return successful / len(self.task_history)
