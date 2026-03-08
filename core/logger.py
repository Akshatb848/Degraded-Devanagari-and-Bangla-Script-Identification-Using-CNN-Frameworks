"""
Inference logger — structured per-request logging with timing metrics.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger("aiocr")


@dataclass
class StepMetric:
    name: str
    duration_ms: float
    status: str          # "ok" | "fallback" | "error"
    detail: str = ""


@dataclass
class RequestLog:
    request_id: str
    filename: str
    started_at: float = field(default_factory=time.time)
    steps: List[StepMetric] = field(default_factory=list)
    script: str = "unknown"
    confidence: float = 0.0
    ocr_chars: int = 0
    llm_model: str = "none"

    # ── helpers ──────────────────────────────────────────────────────────────

    def add_step(self, name: str, duration_ms: float, status: str, detail: str = "") -> None:
        self.steps.append(StepMetric(name, round(duration_ms, 1), status, detail))

    def total_ms(self) -> float:
        return round((time.time() - self.started_at) * 1000, 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "filename": self.filename,
            "total_ms": self.total_ms(),
            "script": self.script,
            "confidence": self.confidence,
            "ocr_chars": self.ocr_chars,
            "llm_model": self.llm_model,
            "steps": [
                {"name": s.name, "ms": s.duration_ms, "status": s.status, "detail": s.detail}
                for s in self.steps
            ],
        }

    def emit(self) -> None:
        log.info("request_complete", **self.as_dict())
