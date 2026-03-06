"""
Full Pipeline API endpoint.
Runs all 7 agents in sequence for complete document processing.
"""
import time
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks

from app.api.models.schemas import (
    PipelineResponse,
    AsyncJobResponse,
    JobStatusResponse,
    ProcessingStatus,
)
from app.core.logging import logger
from agents.orchestrator import AIRCOrchestrator
from workers.celery_app import run_full_pipeline_task
from services.cache_service import CacheService

router = APIRouter(prefix="/full-pipeline", tags=["Full Pipeline"])


@router.post(
    "/",
    response_model=PipelineResponse,
    summary="Run full Agentic OCR pipeline",
    description=(
        "Runs all 7 agents: Script Detection → Image Restoration → "
        "Text Detection → Character Recognition → LLM Correction → "
        "Knowledge Retrieval → Output Formatting"
    ),
)
async def run_full_pipeline(
    file: UploadFile = File(..., description="Document image or PDF"),
    language_hint: Optional[str] = Query(None, description="Language hint for OCR"),
    enable_restoration: bool = Query(True, description="Enable image restoration agent"),
    enable_rag: bool = Query(True, description="Enable knowledge retrieval (RAG)"),
    include_annotated_image: bool = Query(False, description="Include annotated image in response"),
) -> PipelineResponse:
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        "full_pipeline_request",
        request_id=request_id,
        filename=file.filename,
        enable_restoration=enable_restoration,
        enable_rag=enable_rag,
    )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File size must not exceed 10MB")

        orchestrator = AIRCOrchestrator()
        result = await orchestrator.run(
            image_bytes=image_bytes,
            request_id=request_id,
            language_hint=language_hint,
            enable_restoration=enable_restoration,
            enable_rag=enable_rag,
            include_annotated_image=include_annotated_image,
        )

        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = round(processing_time, 2)

        logger.info(
            "full_pipeline_complete",
            request_id=request_id,
            script=result["script"],
            confidence=result["overall_confidence"],
            processing_time_ms=result["processing_time_ms"],
        )

        return PipelineResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("full_pipeline_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@router.post(
    "/async",
    response_model=AsyncJobResponse,
    summary="Run full pipeline asynchronously",
    description="Submits document for async processing. Use /full-pipeline/status/{job_id} to poll results.",
)
async def run_full_pipeline_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language_hint: Optional[str] = Query(None),
    enable_restoration: bool = Query(True),
    enable_rag: bool = Query(True),
) -> AsyncJobResponse:
    job_id = str(uuid.uuid4())

    try:
        image_bytes = await file.read()

        # Submit to Celery task queue
        task = run_full_pipeline_task.apply_async(
            args=[image_bytes, job_id, language_hint, enable_restoration, enable_rag],
            task_id=job_id,
        )

        cache = CacheService()
        await cache.set_job_status(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            metadata={"filename": file.filename},
        )

        logger.info("async_pipeline_submitted", job_id=job_id, celery_task_id=task.id)

        return AsyncJobResponse(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            message="Document submitted for processing",
            result_url=f"/api/v1/full-pipeline/status/{job_id}",
        )

    except Exception as e:
        logger.error("async_pipeline_submission_error", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to submit pipeline job: {str(e)}")


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get async pipeline job status",
)
async def get_pipeline_status(job_id: str) -> JobStatusResponse:
    cache = CacheService()
    job = await cache.get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(**job)
