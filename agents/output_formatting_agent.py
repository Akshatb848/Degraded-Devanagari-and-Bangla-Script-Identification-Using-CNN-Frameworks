"""
Agent 7: Output Formatting Agent
Generates structured JSON output with bounding boxes, confidence scores,
and optional annotated image overlay.
"""
import base64
import io
import time
from typing import List, Dict, Any, Optional
import structlog

from agents.state import PipelineState

logger = structlog.get_logger()


def output_formatting_agent(state: PipelineState) -> PipelineState:
    """
    Assemble the final structured output from all agent results.

    Input:  All agent state fields
    Output: state.final_output, state.annotated_image_base64
    """
    start = time.time()
    request_id = state["request_id"]
    logger.info("agent_start", agent="OutputFormattingAgent", request_id=request_id)

    try:
        # Choose best available text
        final_text = (
            state.get("validated_text")
            or state.get("corrected_text")
            or state.get("raw_text")
            or ""
        )

        # Compute overall confidence
        overall_confidence = _compute_overall_confidence(state)

        # Build text regions with bounding boxes
        text_regions = _build_text_regions(state)
        bounding_boxes = [r["bounding_box"] for r in text_regions]

        # Collect all corrections
        all_corrections = []
        all_corrections.extend(state.get("corrections_made") or [])
        all_corrections.extend(state.get("rag_corrections") or [])

        # Annotated image
        annotated_b64 = None
        if state.get("include_annotated_image") and state.get("text_regions_raw"):
            annotated_b64 = _create_annotated_image(state)

        final_output = {
            "request_id": request_id,
            "status": "completed",
            "script": state.get("script", "unknown"),
            "raw_text": state.get("raw_text", ""),
            "corrected_text": final_text,
            "overall_confidence": overall_confidence,
            "text_regions": text_regions,
            "bounding_boxes": bounding_boxes,
            "language": _script_to_language(state.get("script", "unknown")),
            "reasoning": state.get("reasoning", ""),
            "corrections_made": all_corrections,
            "agent_statuses": _format_agent_statuses(state),
            "restored_image_base64": None,
            "annotated_image_base64": annotated_b64,
        }

        # Add restored image if requested
        if state.get("restored_image_bytes") and state.get("include_annotated_image"):
            final_output["restored_image_base64"] = base64.b64encode(
                state["restored_image_bytes"]
            ).decode("utf-8")

        state["final_output"] = final_output
        state["overall_confidence"] = overall_confidence
        if annotated_b64:
            state["annotated_image_base64"] = annotated_b64

        elapsed = (time.time() - start) * 1000
        _record_agent_status(
            state,
            "OutputFormattingAgent",
            "completed",
            output={"confidence": overall_confidence, "text_regions": len(text_regions)},
            processing_time_ms=elapsed,
        )

        logger.info(
            "agent_complete",
            agent="OutputFormattingAgent",
            request_id=request_id,
            confidence=overall_confidence,
            elapsed_ms=elapsed,
        )

    except Exception as e:
        logger.error("agent_error", agent="OutputFormattingAgent", error=str(e))
        state["final_output"] = _error_output(state, str(e))
        _record_agent_status(state, "OutputFormattingAgent", "failed", error=str(e))
        _add_error(state, f"OutputFormattingAgent: {e}")

    return state


def _compute_overall_confidence(state: PipelineState) -> float:
    """Weighted average of script detection and OCR confidence."""
    script_conf = state.get("script_confidence", 0.0) or 0.0
    ocr_conf = state.get("ocr_confidence", 0.0) or 0.0

    # Weight script detection more heavily if text is short
    text_len = len(state.get("raw_text", "") or "")
    if text_len < 50:
        return round(script_conf * 0.7 + ocr_conf * 0.3, 4)
    return round(script_conf * 0.3 + ocr_conf * 0.7, 4)


def _build_text_regions(state: PipelineState) -> List[Dict[str, Any]]:
    """Build structured text region objects."""
    regions_raw = state.get("text_regions_raw") or []
    texts_per_region = state.get("raw_text_per_region") or []
    corrected_texts = state.get("raw_text_per_region") or []

    # If LLM corrected the whole text, we can't map back to regions precisely
    # so we use the raw per-region texts as corrected too
    result = []
    for i, region in enumerate(regions_raw):
        raw = texts_per_region[i] if i < len(texts_per_region) else ""
        corrected = corrected_texts[i] if i < len(corrected_texts) else raw
        bbox = region.get("bbox", {})

        result.append({
            "bounding_box": {
                "x": bbox.get("x", 0),
                "y": bbox.get("y", 0),
                "width": bbox.get("width", 0),
                "height": bbox.get("height", 0),
                "confidence": region.get("confidence", 0.8),
            },
            "raw_text": raw,
            "corrected_text": corrected,
            "confidence": region.get("confidence", 0.8),
            "line_number": region.get("line_number", i),
        })

    return result


def _create_annotated_image(state: PipelineState) -> Optional[str]:
    """Draw bounding boxes on the image and return as base64."""
    try:
        import cv2
        import numpy as np

        image_bytes = state.get("restored_image_bytes") or state["image_bytes"]
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        regions = state.get("text_regions_raw") or []
        texts = state.get("raw_text_per_region") or []

        for i, region in enumerate(regions):
            bbox = region.get("bbox", {})
            x, y = bbox.get("x", 0), bbox.get("y", 0)
            w, h = bbox.get("width", 0), bbox.get("height", 0)
            conf = region.get("confidence", 0.8)

            # Color: green for high confidence, yellow for medium, red for low
            if conf >= 0.8:
                color = (0, 200, 0)
            elif conf >= 0.5:
                color = (0, 200, 200)
            else:
                color = (0, 0, 200)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Add line number label
            label = f"L{i+1}"
            cv2.putText(img, label, (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        _, buffer = cv2.imencode(".png", img)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    except Exception as e:
        logger.warning("annotation_failed", error=str(e))
        return None


def _script_to_language(script: str) -> str:
    mapping = {
        "devanagari": "Hindi/Sanskrit/Marathi",
        "bangla": "Bengali",
        "unknown": "Unknown",
    }
    return mapping.get(script, "Unknown")


def _format_agent_statuses(state: PipelineState) -> List[Dict[str, Any]]:
    return state.get("agent_statuses") or []


def _error_output(state: PipelineState, error: str) -> Dict[str, Any]:
    return {
        "request_id": state.get("request_id", ""),
        "status": "failed",
        "script": state.get("script", "unknown"),
        "raw_text": state.get("raw_text", ""),
        "corrected_text": state.get("corrected_text", ""),
        "overall_confidence": 0.0,
        "text_regions": [],
        "bounding_boxes": [],
        "language": "Unknown",
        "reasoning": f"Output formatting failed: {error}",
        "corrections_made": [],
        "agent_statuses": state.get("agent_statuses") or [],
        "restored_image_base64": None,
        "annotated_image_base64": None,
    }


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
