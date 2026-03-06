"""
Script Detection API endpoint.
Agent 1: CNN-based script identification (Devanagari vs Bangla).
"""
import time
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.api.models.schemas import ScriptDetectionResponse, ScriptType, ErrorResponse
from app.core.logging import logger
from models.cnn_classifier import ScriptClassifier

router = APIRouter(prefix="/detect-script", tags=["Script Detection"])


@router.post(
    "/",
    response_model=ScriptDetectionResponse,
    summary="Detect script type in an image",
    description="Uses trained CNN models (VGG16/DenseNet/ResNet/AlexNet) to classify Devanagari or Bangla script.",
)
async def detect_script(
    file: UploadFile = File(..., description="Image file (JPG, PNG, TIFF, BMP)"),
    model_name: str = "ensemble",
) -> ScriptDetectionResponse:
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info("script_detection_request", request_id=request_id, filename=file.filename)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, TIFF, BMP)")

    try:
        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image size must not exceed 10MB")

        classifier = ScriptClassifier.get_instance()
        script, confidence, model_used = classifier.predict(image_bytes, model_name=model_name)

        processing_time = (time.time() - start_time) * 1000

        response = ScriptDetectionResponse(
            request_id=request_id,
            script=ScriptType(script),
            confidence=confidence,
            model_used=model_used,
            processing_time_ms=round(processing_time, 2),
        )

        logger.info(
            "script_detection_complete",
            request_id=request_id,
            script=script,
            confidence=confidence,
            processing_time_ms=response.processing_time_ms,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("script_detection_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Script detection failed: {str(e)}")
