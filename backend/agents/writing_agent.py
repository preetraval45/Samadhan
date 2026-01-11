"""
Writing Agent
Specialized in content generation and editing
"""

from typing import List, Dict, Any
from datetime import datetime, timezone
import time
from .base_agent import BaseAgent, AgentTask, AgentResult, AgentCapability
from loguru import logger


class WritingAgent(BaseAgent):
    """
    Agent specialized in writing tasks

    Capabilities:
    - Content generation
    - Text editing and refinement
    - Style adaptation
    - Document structuring
    """

    def __init__(self, llm_engine=None):
        super().__init__(
            name="WritingAgent",
            role="Content Writer",
            description="Creates and refines written content"
        )
        self.llm_engine = llm_engine

    def get_capabilities(self) -> List[AgentCapability]:
        """Define writing agent capabilities"""
        return [
            AgentCapability(
                name="generate_content",
                description="Generate original content on a topic",
                parameters={"topic": "string", "style": "string", "length": "int"}
            ),
            AgentCapability(
                name="edit_text",
                description="Edit and improve existing text",
                parameters={"text": "string", "instructions": "string"}
            ),
            AgentCapability(
                name="adapt_style",
                description="Adapt content to different writing styles",
                parameters={"text": "string", "target_style": "string"}
            ),
            AgentCapability(
                name="structure_document",
                description="Organize content into a structured document",
                parameters={"content": "string", "document_type": "string"}
            )
        ]

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute writing task"""
        start_time = time.time()
        logger.info(f"[{self.name}] Starting task: {task.task_type}")

        try:
            if task.task_type == "generate_content":
                output = await self._generate_content(task.input_data)
            elif task.task_type == "edit_text":
                output = await self._edit_text(task.input_data)
            elif task.task_type == "adapt_style":
                output = await self._adapt_style(task.input_data)
            elif task.task_type == "structure_document":
                output = await self._structure_document(task.input_data)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            execution_time = (time.time() - start_time) * 1000

            result = AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=True,
                output=output,
                metadata={"task_type": task.task_type},
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

    async def _generate_content(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate original content"""
        topic = input_data.get("topic", "")
        style = input_data.get("style", "professional")
        length = input_data.get("length", 500)

        if not self.llm_engine:
            return {"error": "LLM engine not available", "content": ""}

        prompt = f"""Write {length} words about: {topic}

Style: {style}

Requirements:
- Well-structured and coherent
- Engaging and informative
- Appropriate tone for {style} style"""

        response = await self.llm_engine.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=length * 2
        )

        return {
            "topic": topic,
            "content": response["content"],
            "style": style,
            "word_count": len(response["content"].split())
        }

    async def _edit_text(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Edit and improve text"""
        text = input_data.get("text", "")
        instructions = input_data.get("instructions", "improve clarity and readability")

        if not self.llm_engine:
            return {"error": "LLM engine not available", "edited_text": text}

        prompt = f"""Edit the following text according to these instructions: {instructions}

Original text:
{text}

Provide the edited version:"""

        response = await self.llm_engine.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=len(text.split()) * 2
        )

        return {
            "original_text": text,
            "edited_text": response["content"],
            "instructions": instructions
        }

    async def _adapt_style(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt content to different style"""
        text = input_data.get("text", "")
        target_style = input_data.get("target_style", "formal")

        if not self.llm_engine:
            return {"error": "LLM engine not available", "adapted_text": text}

        prompt = f"""Rewrite the following text in a {target_style} style:

{text}

Maintain the core message but adapt the tone, vocabulary, and structure to match the {target_style} style."""

        response = await self.llm_engine.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=len(text.split()) * 2
        )

        return {
            "original_text": text,
            "adapted_text": response["content"],
            "target_style": target_style
        }

    async def _structure_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure content into document format"""
        content = input_data.get("content", "")
        document_type = input_data.get("document_type", "report")

        if not self.llm_engine:
            return {"error": "LLM engine not available", "structured_document": content}

        prompt = f"""Organize the following content into a well-structured {document_type}:

{content}

Create appropriate sections, headings, and flow. Include:
- Title
- Introduction
- Main sections with subheadings
- Conclusion"""

        response = await self.llm_engine.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=len(content.split()) * 3
        )

        return {
            "original_content": content,
            "structured_document": response["content"],
            "document_type": document_type
        }
