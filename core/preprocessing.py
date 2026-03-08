"""
Image Preprocessing Pipeline
─────────────────────────────
Steps (in order):
  1. Decode & validate
  2. Grayscale conversion
  3. Gaussian noise removal
  4. Adaptive thresholding
  5. Morphological closing  (fills ink gaps)
  6. Skew correction
  7. Line segmentation      (bounding-box of each text line)

Returns a PreprocessResult dataclass that carries every intermediate
image so the UI can show before/after comparisons.
"""

import io
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    # Pillow images at each stage (None if step was skipped)
    original:    Image.Image = None
    gray:        Image.Image = None
    denoised:    Image.Image = None
    thresholded: Image.Image = None
    closed:      Image.Image = None
    deskewed:    Image.Image = None

    # Final image ready for OCR/classifier (numpy uint8 BGR)
    final_np: Optional[np.ndarray] = None

    # Line bounding boxes [(x, y, w, h), ...]
    line_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    skew_angle:      float = 0.0
    quality_score:   float = 0.0   # Laplacian variance (sharpness proxy)
    duration_ms:     float = 0.0
    warnings:        List[str] = field(default_factory=list)

    @property
    def final_pil(self) -> Optional[Image.Image]:
        """BGR numpy → RGB PIL for display."""
        if self.final_np is None:
            return None
        return Image.fromarray(cv2.cvtColor(self.final_np, cv2.COLOR_BGR2RGB))


# ── public entry point ────────────────────────────────────────────────────────

def preprocess(image_bytes: bytes) -> PreprocessResult:
    """
    Run the full preprocessing pipeline on raw image bytes.
    Every step is wrapped individually; a failure falls back to
    the previous stage so the pipeline always produces output.
    """
    t_start = time.perf_counter()
    res = PreprocessResult()

    # 1 ── decode ─────────────────────────────────────────────────────────────
    img_bgr = _decode(image_bytes)
    if img_bgr is None:
        res.warnings.append("Could not decode image — returned empty result.")
        return res
    res.original = _bgr_to_pil(img_bgr)

    # 2 ── grayscale ──────────────────────────────────────────────────────────
    gray = _to_gray(img_bgr)
    res.gray = _gray_to_pil(gray)

    # 3 ── Gaussian denoising ─────────────────────────────────────────────────
    denoised = _denoise(gray)
    res.denoised = _gray_to_pil(denoised)

    # 4 ── adaptive thresholding ──────────────────────────────────────────────
    thresh = _threshold(denoised)
    res.thresholded = _gray_to_pil(thresh)

    # 5 ── morphological closing ──────────────────────────────────────────────
    closed = _morph_close(thresh)
    res.closed = _gray_to_pil(closed)

    # 6 ── skew correction ────────────────────────────────────────────────────
    deskewed, angle = _deskew(closed)
    res.deskewed  = _gray_to_pil(deskewed)
    res.skew_angle = angle
    if abs(angle) > 10:
        res.warnings.append(f"Large skew detected ({angle:.1f}°) — result may be imperfect.")

    # 7 ── line segmentation ──────────────────────────────────────────────────
    res.line_bboxes = _segment_lines(deskewed)

    # Final: convert back to BGR for downstream consumers
    res.final_np    = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
    res.quality_score = _laplacian_quality(deskewed)
    res.duration_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return res


# ── step implementations ──────────────────────────────────────────────────────

def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img  # None if decode failed
    except Exception:
        return None


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    try:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        # already single-channel
        return img_bgr if img_bgr.ndim == 2 else img_bgr[:, :, 0]


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Two-pass denoising:
      • Gaussian blur  — fast, removes high-frequency sensor noise
      • Non-local means — preserves text edges better than pure Gaussian
    """
    try:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.fastNlMeansDenoising(blurred, h=10, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        return gray


def _threshold(denoised: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding binarises the image while
    compensating for uneven illumination across the page.
    """
    try:
        return cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
    except Exception:
        _, t = cv2.threshold(denoised, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return t


def _morph_close(thresh: np.ndarray) -> np.ndarray:
    """
    Morphological closing fills small breaks in ink strokes
    (common in aged manuscripts) without merging separate glyphs.
    """
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    except Exception:
        return thresh


def _deskew(binary: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Correct document skew using the minimum area rectangle of
    all foreground pixels.  Returns (corrected_image, angle_degrees).
    """
    try:
        # Invert so text pixels are white (foreground)
        inv = cv2.bitwise_not(binary)
        coords = np.column_stack(np.where(inv > 0))

        if len(coords) < 100:
            return binary, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect angle convention: normalise to (-45, 45]
        if angle < -45:
            angle = 90 + angle
        # Tiny skew — not worth rotating (introduces interpolation noise)
        if abs(angle) < 0.5:
            return binary, 0.0

        h, w = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(
            binary, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,       # white background
        )
        return rotated, round(float(angle), 2)
    except Exception:
        return binary, 0.0


def _segment_lines(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Segment text lines via horizontal projection profile.
    Returns a list of (x, y, width, height) bounding boxes,
    one per detected text line.
    """
    try:
        inv = cv2.bitwise_not(binary)

        # Dilate horizontally to merge characters in the same line
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        dilated = cv2.dilate(inv, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        min_area = binary.shape[1] * 5   # at least 5px tall × full width

        for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area and h >= 8:
                bboxes.append((x, y, w, h))

        return bboxes
    except Exception:
        return []


def _laplacian_quality(gray: np.ndarray) -> float:
    """
    Laplacian variance as a sharpness / quality proxy.
    Normalised to [0, 1] (saturates at variance ≥ 5000).
    """
    try:
        return round(min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 5000.0, 1.0), 4)
    except Exception:
        return 0.0


# ── utilities ─────────────────────────────────────────────────────────────────

def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _gray_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
