"""
Samādhān - Decision Intelligence Platform
Main FastAPI Application Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from api.routes import chat, health, models, documents, tools
from core.config import settings
from core.logging import setup_logging

# Setup logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    logger.info("Starting Samādhān Decision Intelligence Platform...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API Version: {settings.API_VERSION}")

    # Initialize services here
    # await initialize_vector_store()
    # await initialize_llm_engine()

    yield

    logger.info("Shutting down Samādhān...")
    # Cleanup here
    # await cleanup_connections()


# Initialize FastAPI app
app = FastAPI(
    title="Samādhān API",
    description="Multi-Modal AI-Powered Research & Decision Intelligence Platform",
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with platform information"""
    return {
        "platform": "Samādhān",
        "tagline": "Decision Intelligence Platform",
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/api/docs"
    }


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(tools.router, prefix="/api/v1", tags=["Tools"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
