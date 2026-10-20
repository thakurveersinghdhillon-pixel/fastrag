"""FastRAG FastAPI Application Entry Point.

Phase 1 establishes a lightweight ASGI application with health monitoring,
centralized settings, and structured logging.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.v1.routes import router as api_v1_router
from app.core.config import Settings, get_settings
from app.core.logging_config import setup_logging, get_logger

logger = get_logger("fastrag.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lightweight application lifespan management with resource pre-warming and cleanup."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        f"Starting {settings.app_name} in '{settings.environment}' mode",
        extra={"operation": "startup"},
    )
    from app.api.dependencies import create_pipeline

    is_test_env = settings.environment.lower() == "test"
    pipeline = create_pipeline(settings, warmup=not is_test_env)
    app.state.pipeline = pipeline

    yield

    logger.info(
        f"Shutting down {settings.app_name}",
        extra={"operation": "shutdown"},
    )
    if hasattr(app.state, "pipeline") and app.state.pipeline is not None:
        await app.state.pipeline.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory for FastRAG FastAPI service."""
    current_settings = settings or get_settings()
    setup_logging(current_settings.log_level)

    app = FastAPI(
        title=current_settings.app_name,
        description="Performance-Oriented Retrieval-Augmented Generation System",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["Health"])
    async def health_check() -> JSONResponse:
        """Health check endpoint providing service status and active environment."""
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "app": current_settings.app_name,
                "environment": current_settings.environment,
                "version": "0.1.0",
            },
        )

    app.include_router(api_v1_router, prefix=current_settings.api_v1_prefix)

    return app


app = create_app()
