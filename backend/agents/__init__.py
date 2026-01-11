"""
Multi-Agent System
Coordinated AI agents for complex task execution
"""

from .base_agent import BaseAgent, AgentTask, AgentResult, AgentCapability
from .research_agent import ResearchAgent
from .writing_agent import WritingAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    "BaseAgent",
    "AgentTask",
    "AgentResult",
    "AgentCapability",
    "ResearchAgent",
    "WritingAgent",
    "AgentOrchestrator"
]
