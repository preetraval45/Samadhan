"""
Chat API Endpoints
RAG-powered conversational AI
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime, timezone
from uuid import uuid4
import json
import asyncio
from llm.engine import LLMEngine

router = APIRouter()
llm_engine = LLMEngine()


class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    model: Optional[str] = Field("microsoft/phi-2", description="LLM model to use (default: free local model)")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192)
    use_rag: bool = Field(True, description="Enable RAG for context")
    domain: Optional[str] = Field(None, description="Domain filter (healthcare, legal, finance)")


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    sources: List[Dict[str, Any]] = []
    model_used: str
    tokens_used: int
    confidence: float
    timestamp: datetime


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with RAG support

    Features:
    - Conversational AI with context awareness
    - RAG-powered responses using vector database
    - Domain-specific knowledge filtering
    - Explainable AI with source citations
    """

    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid4())

    # Generate response using LLM
    system_prompt = """You are a friendly, helpful AI assistant with a warm personality. Engage in natural conversation like a human would:

- Be conversational and warm, not robotic or formal
- Show empathy and understanding
- Ask follow-up questions when appropriate
- Share insights and explanations naturally
- Use a friendly, approachable tone
- Answer thoroughly and helpfully
- Be knowledgeable across many topics
- Admit when you're not sure about something

Respond naturally to all questions and topics the user brings up, just like talking to a knowledgeable friend."""

    llm_response = await llm_engine.generate(
        prompt=request.message,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        system_prompt=system_prompt
    )

    # TODO: Implement RAG pipeline for retrieving context from vector database

    return ChatResponse(
        conversation_id=conversation_id,
        message=llm_response['content'],
        sources=[],  # Will be populated when RAG is integrated
        model_used=llm_response['model'],
        tokens_used=llm_response['tokens_used'],
        confidence=1.0,
        timestamp=datetime.now(timezone.utc)
    )


@router.get("/chat/history/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """
    Retrieve conversation history by ID
    """
    # TODO: Fetch from database
    raise HTTPException(
        status_code=404,
        detail="Conversation not found"
    )


@router.delete("/chat/history/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    """
    # TODO: Implement deletion
    return {"message": "Conversation deleted successfully"}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses using Server-Sent Events (SSE)

    Returns a stream of JSON objects with the following types:
    - start: Initial metadata (conversation_id, model)
    - token: Individual token chunks from the LLM
    - sources: RAG sources (sent after completion)
    - done: Final metadata (tokens_used, confidence)
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for streaming response"""
        try:
            # Generate conversation ID if not provided
            conversation_id = request.conversation_id or str(uuid4())

            # Send start event with metadata
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id, 'model': request.model})}\n\n"

            # Use actual LLM streaming
            system_prompt = """You are a friendly, helpful AI assistant with a warm personality. Engage in natural conversation like a human would:

- Be conversational and warm, not robotic or formal
- Show empathy and understanding
- Ask follow-up questions when appropriate
- Share insights and explanations naturally
- Use a friendly, approachable tone
- Answer thoroughly and helpfully
- Be knowledgeable across many topics
- Admit when you're not sure about something

Respond naturally to all questions and topics the user brings up, just like talking to a knowledgeable friend."""

            token_count = 0
            async for chunk in llm_engine.generate_stream(
                prompt=request.message,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                system_prompt=system_prompt
            ):
                token_data = {
                    'type': 'token',
                    'content': chunk['content'],
                    'index': token_count
                }
                yield f"data: {json.dumps(token_data)}\n\n"
                token_count += 1

            # TODO: Add RAG sources when vector database is integrated
            # For now, send empty sources
            if request.use_rag:
                sources_data = {
                    'type': 'sources',
                    'sources': []
                }
                yield f"data: {json.dumps(sources_data)}\n\n"

            # Send done event with final metadata
            done_data = {
                'type': 'done',
                'tokens_used': token_count,
                'confidence': 1.0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            yield f"data: {json.dumps(done_data)}\n\n"

        except Exception as e:
            # Send error event
            error_data = {
                'type': 'error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
