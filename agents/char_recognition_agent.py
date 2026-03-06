"""
Agent 4: Character Recognition Agent
Uses TrOCR (Vision Transformer OCR) for sequence recognition on detected text regions.
Falls back to Tesseract for Indic scripts if TrOCR unavailable.
"""
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import structlog

from agents.state import PipelineState

logger = structlog.get_logger()

# Mapping from script to Tesseract language codes
TESSERACT_LANG_MAP = {
    "devanagari": "hin+san",
    "bangla": "ben",
    "unknown": "hin+ben",
}


def char_recognition_agent(state: PipelineState) -> PipelineState:
    """
    Perform OCR on each detected text region.

    Input:  state.text_regions_raw
    Output: state.raw_text, state.raw_text_per_region, state.ocr_confidence
    """
    start = time.time()
    request_id = state["request_id"]
    logger.info("agent_start", agent="CharRecognitionAgent", request_id=request_id)

    try:
        regions = state.get("text_regions_raw") or []
        script = state.get("script", "unknown")

        if not regions:
            state["raw_text"] = ""
            state["raw_text_per_region"] = []
            state["ocr_confidence"] = 0.0
            _record_agent_status(state, "CharRecognitionAgent", "no_regions")
            return state

        texts_per_region, ocr_method = _run_ocr_on_regions(regions, script)

        # Combine text from all regions (sorted by line number / y position)
        raw_text = "\n".join(t for t in texts_per_region if t.strip())
        avg_confidence = _estimate_confidence(texts_per_region, regions)

        elapsed = (time.time() - start) * 1000

        state["raw_text"] = raw_text
        state["raw_text_per_region"] = texts_per_region
        state["ocr_confidence"] = avg_confidence

        _record_agent_status(
            state,
            "CharRecognitionAgent",
            "completed",
            output={
                "regions_processed": len(regions),
                "method": ocr_method,
                "avg_confidence": avg_confidence,
                "text_length": len(raw_text),
            },
            processing_time_ms=elapsed,
        )

        logger.info(
            "agent_complete",
            agent="CharRecognitionAgent",
            request_id=request_id,
            method=ocr_method,
            regions=len(regions),
            text_len=len(raw_text),
            elapsed_ms=elapsed,
        )

    except Exception as e:
        logger.error("agent_error", agent="CharRecognitionAgent", error=str(e))
        state["raw_text"] = ""
        state["raw_text_per_region"] = []
        state["ocr_confidence"] = 0.0
        _record_agent_status(state, "CharRecognitionAgent", "failed", error=str(e))
        _add_error(state, f"CharRecognitionAgent: {e}")

    return state


def _run_ocr_on_regions(
    regions: List[Dict[str, Any]],
    script: str,
) -> Tuple[List[str], str]:
    """
    Try TrOCR first; fall back to Tesseract.
    """
    try:
        return _trocr_ocr(regions, script), "trocr"
    except Exception as e:
        logger.warning("trocr_failed", error=str(e))

    try:
        return _tesseract_ocr(regions, script), "tesseract"
    except Exception as e:
        logger.warning("tesseract_failed", error=str(e))

    return [""] * len(regions), "failed"


def _trocr_ocr(regions: List[Dict[str, Any]], script: str) -> List[str]:
    """
    TrOCR-based character recognition.
    Uses microsoft/trocr-base-handwritten as base model.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch
    from app.core.config import settings

    model_path = settings.trocr_model_path
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    texts = []
    for region in regions:
        try:
            pil_image = Image.open(__import__("io").BytesIO(region["region_image_bytes"])).convert("RGB")
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            texts.append(text.strip())
        except Exception:
            texts.append("")

    return texts


def _tesseract_ocr(regions: List[Dict[str, Any]], script: str) -> List[str]:
    """
    Tesseract-based OCR fallback for Indic scripts.
    Requires tesseract-ocr with hin/ben language packs.
    """
    import pytesseract
    from PIL import Image
    import io as _io

    lang = TESSERACT_LANG_MAP.get(script, "hin+ben")
    config = f"--oem 3 --psm 6 -l {lang}"
    texts = []

    for region in regions:
        try:
            pil_image = Image.open(_io.BytesIO(region["region_image_bytes"])).convert("RGB")
            text = pytesseract.image_to_string(pil_image, config=config)
            texts.append(text.strip())
        except Exception:
            texts.append("")

    return texts


def _estimate_confidence(texts: List[str], regions: List[Dict[str, Any]]) -> float:
    """
    Estimate overall OCR confidence.
    Combines region detection confidence with text quality heuristics.
    """
    if not texts:
        return 0.0

    detection_confs = [r.get("confidence", 0.8) for r in regions]
    avg_detection = float(np.mean(detection_confs)) if detection_confs else 0.8

    # Text quality: penalize very short or empty regions
    non_empty = sum(1 for t in texts if len(t.strip()) > 0)
    text_ratio = non_empty / len(texts) if texts else 0.0

    return round(avg_detection * text_ratio, 4)


def _record_agent_status(state, agent_name, status, output=None, error=None, processing_time_ms=None):
    if state.get("agent_statuses") is None:
        state["agent_statuses"] = []
    state["agent_statuses"].append({
        "agent_name": agent_name,
        "status": status,
        "output": output,
        "error": error,
        "processing_time_ms": processing_time_ms,
    })


def _add_error(state, error):
    if state.get("errors") is None:
        state["errors"] = []
    state["errors"].append(error)
