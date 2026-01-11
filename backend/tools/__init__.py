"""
AI Tools Module
External tools that AI can use to enhance responses
"""

from .web_search import WebSearchTool, WEB_SEARCH_TOOL_SCHEMA

__all__ = [
    "WebSearchTool",
    "WEB_SEARCH_TOOL_SCHEMA"
]
