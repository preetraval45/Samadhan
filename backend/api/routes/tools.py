"""
Tools API Endpoints
External tools integration for AI
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from tools.web_search import WebSearchTool
from llm.engine import LLMEngine

router = APIRouter()
web_search_tool = WebSearchTool()
llm_engine = LLMEngine()


class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=10, description="Maximum results")
    provider: str = Field("duckduckgo", description="Search provider")
    summarize: bool = Field(True, description="Generate AI summary")


class WebSearchResponse(BaseModel):
    query: str
    provider: str
    results: List[Dict[str, Any]]
    total_results: int
    summary: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: str


@router.post("/tools/web-search", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest):
    """
    Search the web for real-time information

    Features:
    - Multiple search providers (DuckDuckGo, Google)
    - Optional AI-powered summarization
    - Source citations
    """
    try:
        if request.summarize:
            # Search and generate summary
            result = await web_search_tool.search_and_summarize(
                query=request.query,
                llm_engine=llm_engine,
                max_results=request.max_results
            )
        else:
            # Search only
            result = await web_search_tool.search(
                query=request.query,
                max_results=request.max_results,
                provider=request.provider
            )

        return WebSearchResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Web search failed: {str(e)}"
        )


@router.get("/tools/available")
async def get_available_tools():
    """
    List all available AI tools
    """
    from tools import WEB_SEARCH_TOOL_SCHEMA

    return {
        "tools": [
            WEB_SEARCH_TOOL_SCHEMA,
            {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "status": "planned"
            },
            {
                "name": "code_executor",
                "description": "Execute Python code in a sandbox",
                "status": "planned"
            },
            {
                "name": "image_analysis",
                "description": "Analyze images and extract information",
                "status": "planned"
            }
        ],
        "total": 4,
        "implemented": 1
    }


@router.on_event("shutdown")
async def shutdown_tools():
    """Cleanup on shutdown"""
    await web_search_tool.close()
