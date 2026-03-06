"""
Application configuration using Pydantic Settings.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Application
    app_name: str = "AIOCR - Agentic Indic OCR Platform"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Security
    secret_key: str = "change-me-in-production"
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = ["*"]

    # Database
    database_url: str = "postgresql://aiocr:aiocr_password@localhost:5432/aiocr_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # LLM
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Vector DB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "indic_corpus"

    # AWS S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket: str = "aiocr-documents"

    # Model Paths
    model_dir: str = "/app/saved_models"
    cnn_model_path: str = "/app/saved_models/script_classifier.keras"
    yolo_model_path: str = "/app/saved_models/text_detector.pt"
    trocr_model_path: str = "microsoft/trocr-base-handwritten"
    esrgan_model_path: str = "/app/saved_models/esrgan.pth"

    # Image Processing
    image_target_size: int = 64
    max_image_size_mb: int = 10
    supported_formats: List[str] = ["jpg", "jpeg", "png", "tiff", "bmp", "pdf"]

    # Monitoring
    sentry_dsn: Optional[str] = None
    log_level: str = "INFO"

    # Worker
    celery_task_soft_time_limit: int = 300
    celery_task_time_limit: int = 600
    max_concurrent_tasks: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
