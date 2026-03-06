"""
Image Restoration API endpoint.
Agent 2: ESRGAN/SwinIR-based degradation restoration.
"""
import time
import uuid
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.api.models.schemas import ImageRestorationResponse, ErrorResponse
from app.core.logging import logger
from services.image_service import ImageRestorationService

router = APIRouter(prefix="/restore-image", tags=["Image Restoration"])


@router.post(
    "/",
    response_model=ImageRestorationResponse,
    summary="Restore degraded document image",
    description="Enhances degraded images using super-resolution and denoising (ESRGAN/SwinIR).",
)
async def restore_image(
    file: UploadFile = File(..., description="Degraded image file"),
    enhance_resolution: bool = Query(True, description="Apply super-resolution"),
    denoise: bool = Query(True, description="Apply denoising"),
    deskew: bool = Query(True, description="Apply deskewing"),
    return_base64: bool = Query(False, description="Return restored image as base64"),
) -> ImageRestorationResponse:
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info("image_restoration_request", request_id=request_id, filename=file.filename)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()

        service = ImageRestorationService()
        result = await service.restore(
            image_bytes=image_bytes,
            enhance_resolution=enhance_resolution,
            denoise=denoise,
            deskew=deskew,
        )

        processing_time = (time.time() - start_time) * 1000

        restored_b64 = None
        if return_base64:
            restored_b64 = base64.b64encode(result["restored_bytes"]).decode("utf-8")

        response = ImageRestorationResponse(
            request_id=request_id,
            restored_image_base64=restored_b64,
            enhancement_applied=result["enhancements"],
            quality_score_before=result["quality_before"],
            quality_score_after=result["quality_after"],
            processing_time_ms=round(processing_time, 2),
        )

        logger.info(
            "image_restoration_complete",
            request_id=request_id,
            quality_improvement=result["quality_after"] - result["quality_before"],
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("image_restoration_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Image restoration failed: {str(e)}")
