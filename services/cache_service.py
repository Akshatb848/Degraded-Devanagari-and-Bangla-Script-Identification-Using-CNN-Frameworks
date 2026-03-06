"""
Redis-based cache and job status service.
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger()

JOB_TTL_SECONDS = 3600  # 1 hour


class CacheService:
    """Manages job state and caching via Redis."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            from app.core.config import settings
            self._client = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._client

    async def set_job_status(
        self,
        job_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        progress: Optional[int] = None,
    ) -> None:
        client = self._get_client()
        now = datetime.utcnow().isoformat()
        key = f"aiocr:job:{job_id}"

        existing_raw = await client.get(key)
        existing = json.loads(existing_raw) if existing_raw else {}

        job_data = {
            "job_id": job_id,
            "status": status,
            "result": result,
            "error": error,
            "progress_percent": progress,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            **(metadata or {}),
        }

        await client.setex(key, JOB_TTL_SECONDS, json.dumps(job_data))

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        key = f"aiocr:job:{job_id}"
        raw = await client.get(key)
        if raw:
            return json.loads(raw)
        return None

    async def cache_result(self, cache_key: str, result: Dict[str, Any], ttl: int = 300) -> None:
        client = self._get_client()
        key = f"aiocr:cache:{cache_key}"
        await client.setex(key, ttl, json.dumps(result))

    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        key = f"aiocr:cache:{cache_key}"
        raw = await client.get(key)
        if raw:
            return json.loads(raw)
        return None
