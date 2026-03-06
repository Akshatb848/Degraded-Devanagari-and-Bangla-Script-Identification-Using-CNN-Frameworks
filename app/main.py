"""
AIOCR - Agentic Indic OCR Platform
FastAPI application entry point.
"""
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.logging import configure_logging
from app.api.models.schemas import HealthResponse

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    configure_logging()
    logger.info("aiocr_startup", env=settings.app_env, version="1.0.0")

    # Pre-load CNN models at startup
    try:
        from models.cnn_classifier import ScriptClassifier
        ScriptClassifier.get_instance()
        logger.info("cnn_models_loaded")
    except Exception as e:
        logger.warning("cnn_model_load_warning", error=str(e))

    yield

    logger.info("aiocr_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Multi-Agent AI OCR Platform for Indic Scripts.\n\n"
            "Processes degraded documents through a 7-agent pipeline:\n"
            "Script Detection → Image Restoration → Text Detection → "
            "Character Recognition → LLM Correction → Knowledge Retrieval → "
            "Output Formatting"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
        return response

    # Register routers
    from app.api.routes.detect_script import router as detect_script_router
    from app.api.routes.restore_image import router as restore_image_router
    from app.api.routes.ocr import router as ocr_router
    from app.api.routes.pipeline import router as pipeline_router

    prefix = settings.api_v1_prefix
    app.include_router(detect_script_router, prefix=prefix)
    app.include_router(restore_image_router, prefix=prefix)
    app.include_router(ocr_router, prefix=prefix)
    app.include_router(pipeline_router, prefix=prefix)

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        from models.cnn_classifier import ScriptClassifier
        models_status = {
            "cnn_classifier": ScriptClassifier.is_loaded(),
        }
        return HealthResponse(models_loaded=models_status)

    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, str]:
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
    )
