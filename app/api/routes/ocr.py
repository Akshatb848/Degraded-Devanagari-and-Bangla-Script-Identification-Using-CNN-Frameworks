"""
OCR API endpoint.
Agents 3 & 4: Text detection + character recognition pipeline.
"""
import time
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.api.models.schemas import OCRResponse, ScriptType
from app.core.logging import logger
from agents.orchestrator import run_ocr_pipeline

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post(
    "/",
    response_model=OCRResponse,
    summary="Perform OCR on a document image",
    description="Detects text regions and recognizes characters using TrOCR/Donut with LLM correction.",
)
async def perform_ocr(
    file: UploadFile = File(..., description="Document image file"),
    script_hint: Optional[str] = Query(None, description="Script hint: 'devanagari' or 'bangla'"),
    apply_correction: bool = Query(True, description="Apply LLM correction to OCR output"),
    return_bboxes: bool = Query(True, description="Return bounding boxes for detected text"),
) -> OCRResponse:
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info("ocr_request", request_id=request_id, filename=file.filename)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()

        result = await run_ocr_pipeline(
            image_bytes=image_bytes,
            script_hint=script_hint,
            apply_correction=apply_correction,
            include_bboxes=return_bboxes,
        )

        processing_time = (time.time() - start_time) * 1000

        response = OCRResponse(
            request_id=request_id,
            script=ScriptType(result["script"]),
            raw_text=result["raw_text"],
            corrected_text=result["corrected_text"],
            confidence=result["confidence"],
            text_regions=result["text_regions"],
            bounding_boxes=result["bounding_boxes"],
            language_detected=result["language"],
            processing_time_ms=round(processing_time, 2),
        )

        logger.info(
            "ocr_complete",
            request_id=request_id,
            script=result["script"],
            confidence=result["confidence"],
            text_length=len(result["corrected_text"]),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("ocr_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


# Import Optional after defining the route to avoid circular imports
from typing import Optional
