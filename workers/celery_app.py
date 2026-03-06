"""
Celery worker configuration and task definitions.
Handles async document processing tasks.
"""
import asyncio
from celery import Celery
from celery.utils.log import get_task_logger
from app.core.config import settings

logger = get_task_logger(__name__)

# Initialize Celery
celery_app = Celery(
    "aiocr",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["workers.celery_app"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,
    task_routes={
        "workers.celery_app.run_full_pipeline_task": {"queue": "pipeline"},
        "workers.celery_app.run_script_detection_task": {"queue": "detection"},
    },
)


@celery_app.task(
    name="workers.celery_app.run_full_pipeline_task",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def run_full_pipeline_task(
    self,
    image_bytes: bytes,
    job_id: str,
    language_hint: str = None,
    enable_restoration: bool = True,
    enable_rag: bool = True,
) -> dict:
    """
    Celery task: Run the full 7-agent AIOCR pipeline.
    Called asynchronously from the /full-pipeline/async endpoint.
    """
    from services.cache_service import CacheService

    async def _run():
        cache = CacheService()
        await cache.set_job_status(job_id=job_id, status="processing", progress=10)

        try:
            from agents.orchestrator import AIRCOrchestrator
            orchestrator = AIRCOrchestrator()

            await cache.set_job_status(job_id=job_id, status="processing", progress=30)

            result = await orchestrator.run(
                image_bytes=image_bytes,
                request_id=job_id,
                language_hint=language_hint,
                enable_restoration=enable_restoration,
                enable_rag=enable_rag,
                include_annotated_image=False,
            )

            await cache.set_job_status(
                job_id=job_id,
                status="completed",
                result=result,
                progress=100,
            )

            logger.info(f"Pipeline task completed: job_id={job_id}")
            return result

        except Exception as e:
            logger.error(f"Pipeline task failed: job_id={job_id}, error={e}")
            await cache.set_job_status(
                job_id=job_id,
                status="failed",
                error=str(e),
                progress=0,
            )
            raise

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    except Exception as exc:
        raise self.retry(exc=exc)
    finally:
        loop.close()


@celery_app.task(
    name="workers.celery_app.run_script_detection_task",
    bind=True,
)
def run_script_detection_task(self, image_bytes: bytes) -> dict:
    """Lightweight task for script detection only."""
    from models.cnn_classifier import ScriptClassifier
    classifier = ScriptClassifier.get_instance()
    script, confidence, model_used = classifier.predict(image_bytes)
    return {
        "script": script,
        "confidence": confidence,
        "model_used": model_used,
    }
