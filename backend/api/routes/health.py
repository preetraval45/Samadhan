"""
Health Check Endpoints
System status and monitoring
"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime
from core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes
    """
    # Add checks for database, vector store, etc.
    return {
        "ready": True,
        "services": {
            "api": "ready",
            "database": "ready",
            "vector_store": "ready",
            "llm_engine": "ready"
        }
    }


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for Kubernetes
    """
    return {"alive": True}
