"""
Agent 3: Text Detection Agent
Uses YOLOv8 (or CRAFT fallback) to detect text regions and return bounding boxes.
"""
import io
import time
from typing import List, Dict, Any, Optional
import numpy as np
import structlog

from agents.state import PipelineState

logger = structlog.get_logger()


def text_detection_agent(state: PipelineState) -> PipelineState:
    """
    Detect text regions in the (restored) document image.

    Input:  state.restored_image_bytes (or state.image_bytes)
    Output: state.text_regions_raw  — list of {bbox, region_image_bytes}
    """
    start = time.time()
    request_id = state["request_id"]
    logger.info("agent_start", agent="TextDetectionAgent", request_id=request_id)

    try:
        # Use restored image if available, otherwise original
        image_bytes = state.get("restored_image_bytes") or state["image_bytes"]

        regions, method = _detect_text_regions(image_bytes)

        elapsed = (time.time() - start) * 1000
        state["text_regions_raw"] = regions

        _record_agent_status(
            state,
            "TextDetectionAgent",
            "completed",
            output={"regions_found": len(regions), "method": method},
            processing_time_ms=elapsed,
        )

        logger.info(
            "agent_complete",
            agent="TextDetectionAgent",
            request_id=request_id,
            regions_found=len(regions),
            method=method,
            elapsed_ms=elapsed,
        )

    except Exception as e:
        logger.error("agent_error", agent="TextDetectionAgent", error=str(e))
        state["text_regions_raw"] = _fallback_full_image(state)
        _record_agent_status(state, "TextDetectionAgent", "fallback", error=str(e))
        _add_error(state, f"TextDetectionAgent: {e}")

    return state


def _detect_text_regions(image_bytes: bytes) -> tuple[List[Dict[str, Any]], str]:
    """
    Try YOLOv8 first, then fall back to CRAFT, then to full-image region.
    Returns (regions, method_name).
    """
    # Try YOLOv8
    try:
        return _yolo_detect(image_bytes), "yolov8"
    except Exception as e:
        logger.warning("yolo_detection_failed", error=str(e))

    # Try CRAFT (text detection model)
    try:
        return _craft_detect(image_bytes), "craft"
    except Exception as e:
        logger.warning("craft_detection_failed", error=str(e))

    # Fallback: connected components
    return _connected_components_detect(image_bytes), "connected_components"


def _yolo_detect(image_bytes: bytes) -> List[Dict[str, Any]]:
    """YOLOv8-based text region detection."""
    from ultralytics import YOLO
    from app.core.config import settings
    import cv2

    model_path = settings.yolo_model_path
    model = YOLO(model_path)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.25, iou=0.45)
    regions = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            region_img = img[y1:y2, x1:x2]
            _, buf = cv2.imencode(".png", region_img)
            regions.append({
                "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "confidence": conf,
                "region_image_bytes": buf.tobytes(),
                "line_number": len(regions),
            })

    return sorted(regions, key=lambda r: r["bbox"]["y"])


def _craft_detect(image_bytes: bytes) -> List[Dict[str, Any]]:
    """CRAFT-based text detection via DocTR."""
    from doctr.io import DocumentFile
    from doctr.models import detection_predictor

    model = detection_predictor(arch="db_resnet50", pretrained=True)
    doc = DocumentFile.from_images([image_bytes])
    result = model(doc)

    import cv2
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    regions = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                (x0, y0), (x1, y1) = line.geometry
                x, y = int(x0 * w), int(y0 * h)
                bw, bh = int((x1 - x0) * w), int((y1 - y0) * h)
                region_img = img[y:y+bh, x:x+bw]
                _, buf = cv2.imencode(".png", region_img)
                regions.append({
                    "bbox": {"x": x, "y": y, "width": bw, "height": bh},
                    "confidence": 0.9,
                    "region_image_bytes": buf.tobytes(),
                    "line_number": len(regions),
                })

    return regions


def _connected_components_detect(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Fallback text detection using connected components analysis.
    Groups nearby components into text line regions.
    """
    import cv2

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate to merge characters into words/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    min_area = 500  # Filter noise

    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < min_area:
            continue
        region_img = img[y:y+bh, x:x+bw]
        _, buf = cv2.imencode(".png", region_img)
        regions.append({
            "bbox": {"x": x, "y": y, "width": bw, "height": bh},
            "confidence": 0.7,
            "region_image_bytes": buf.tobytes(),
            "line_number": len(regions),
        })

    return regions


def _fallback_full_image(state: PipelineState) -> List[Dict[str, Any]]:
    """Return whole image as single text region."""
    import cv2
    import numpy as np

    image_bytes = state.get("restored_image_bytes") or state["image_bytes"]
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    return [{
        "bbox": {"x": 0, "y": 0, "width": w, "height": h},
        "confidence": 0.5,
        "region_image_bytes": image_bytes,
        "line_number": 0,
    }]


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
