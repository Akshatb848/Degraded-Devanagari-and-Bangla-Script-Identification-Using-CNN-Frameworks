"""
Pydantic schemas for API request/response models.
"""
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class ScriptType(str, Enum):
    DEVANAGARI = "devanagari"
    BANGLA = "bangla"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left x coordinate")
    y: int = Field(..., description="Top-left y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")
    confidence: float = Field(..., ge=0.0, le=1.0)


class TextRegion(BaseModel):
    bounding_box: BoundingBox
    raw_text: str
    corrected_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    line_number: int


class ScriptDetectionResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    script: ScriptType
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ImageRestorationResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    restored_image_url: Optional[str] = None
    restored_image_base64: Optional[str] = None
    enhancement_applied: List[str]
    quality_score_before: float
    quality_score_after: float
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OCRResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    script: ScriptType
    raw_text: str
    corrected_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    text_regions: List[TextRegion]
    bounding_boxes: List[BoundingBox]
    language_detected: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentStatus(BaseModel):
    agent_name: str
    status: ProcessingStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None


class PipelineResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ProcessingStatus
    script: ScriptType
    raw_text: str
    corrected_text: str
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    text_regions: List[TextRegion]
    bounding_boxes: List[BoundingBox]
    language: str
    reasoning: str
    corrections_made: List[str]
    agent_statuses: List[AgentStatus]
    restored_image_base64: Optional[str] = None
    annotated_image_base64: Optional[str] = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AsyncJobResponse(BaseModel):
    job_id: str
    status: ProcessingStatus
    message: str
    result_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobStatusResponse(BaseModel):
    job_id: str
    status: ProcessingStatus
    progress_percent: Optional[int] = None
    result: Optional[PipelineResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
