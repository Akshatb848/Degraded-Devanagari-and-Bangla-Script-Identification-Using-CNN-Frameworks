"""
Agent 2: Image Restoration Agent
Enhances degraded images using ESRGAN and classical CV methods.

Techniques applied:
- Denoising (bilateral filter / non-local means)
- Super-resolution (ESRGAN / bicubic upscale)
- Deskewing
- Contrast enhancement (CLAHE)
- Binarization (adaptive threshold)
"""
import io
import time
from typing import List, Tuple
import numpy as np
import structlog

from agents.state import PipelineState

logger = structlog.get_logger()


def image_restoration_agent(state: PipelineState) -> PipelineState:
    """
    Restore and enhance the degraded document image.

    Input:  state.image_bytes
    Output: state.restored_image_bytes, state.restoration_applied
    """
    if not state.get("enable_restoration", True):
        logger.info("agent_skip", agent="ImageRestorationAgent", reason="disabled")
        state["restored_image_bytes"] = state["image_bytes"]
        state["restoration_applied"] = []
        state["quality_before"] = 0.0
        state["quality_after"] = 0.0
        _record_agent_status(state, "ImageRestorationAgent", "skipped")
        return state

    start = time.time()
    request_id = state["request_id"]
    logger.info("agent_start", agent="ImageRestorationAgent", request_id=request_id)

    try:
        import cv2
        from PIL import Image

        # Load image
        image_bytes = state["image_bytes"]
        np_img = _bytes_to_cv2(image_bytes)

        quality_before = _compute_quality_score(np_img)
        enhancements: List[str] = []

        # Step 1: Deskew
        deskewed = _deskew(np_img)
        if deskewed is not None:
            np_img = deskewed
            enhancements.append("deskew")

        # Step 2: Denoise
        if len(np_img.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(np_img, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(np_img, None, 10, 7, 21)
        np_img = denoised
        enhancements.append("denoising")

        # Step 3: CLAHE contrast enhancement
        np_img = _apply_clahe(np_img)
        enhancements.append("clahe_contrast")

        # Step 4: ESRGAN super-resolution (if model available, else bicubic)
        np_img, sr_method = _apply_super_resolution(np_img)
        enhancements.append(sr_method)

        quality_after = _compute_quality_score(np_img)
        elapsed = (time.time() - start) * 1000

        restored_bytes = _cv2_to_bytes(np_img)

        state["restored_image_bytes"] = restored_bytes
        state["restoration_applied"] = enhancements
        state["quality_before"] = quality_before
        state["quality_after"] = quality_after

        _record_agent_status(
            state,
            "ImageRestorationAgent",
            "completed",
            output={
                "enhancements": enhancements,
                "quality_before": quality_before,
                "quality_after": quality_after,
            },
            processing_time_ms=elapsed,
        )

        logger.info(
            "agent_complete",
            agent="ImageRestorationAgent",
            request_id=request_id,
            enhancements=enhancements,
            quality_delta=quality_after - quality_before,
            elapsed_ms=elapsed,
        )

    except Exception as e:
        logger.error("agent_error", agent="ImageRestorationAgent", error=str(e))
        state["restored_image_bytes"] = state["image_bytes"]
        state["restoration_applied"] = []
        state["quality_before"] = 0.0
        state["quality_after"] = 0.0
        _record_agent_status(state, "ImageRestorationAgent", "failed", error=str(e))
        _add_error(state, f"ImageRestorationAgent: {e}")

    return state


def _bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    import cv2
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _cv2_to_bytes(img: np.ndarray) -> bytes:
    import cv2
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()


def _compute_quality_score(img: np.ndarray) -> float:
    """
    Estimate image quality using Laplacian variance (sharpness proxy).
    Returns a score in [0, 1].
    """
    import cv2
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: assume max useful variance ~5000
    return min(float(variance) / 5000.0, 1.0)


def _deskew(img: np.ndarray):
    """Correct document skew using Hough transform."""
    import cv2
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return None
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:  # negligible skew
            return None

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return None


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast limited adaptive histogram equalization."""
    import cv2
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge((l_ch, a_ch, b_ch))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)


def _apply_super_resolution(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Try ESRGAN super-resolution; fall back to bicubic upscaling.
    Returns (enhanced_image, method_name).
    """
    import cv2
    from app.core.config import settings

    esrgan_path = settings.esrgan_model_path
    if esrgan_path and __import__("os").path.exists(esrgan_path):
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(scale=4, model_path=esrgan_path, model=model, tile=400, half=False)
            output, _ = upsampler.enhance(img, outscale=2)
            return output, "esrgan_4x"
        except Exception:
            pass

    # Fallback: bicubic x2 upscale
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return upscaled, "bicubic_2x"


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
