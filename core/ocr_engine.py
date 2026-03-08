"""
OCR Engine
──────────
Runs Tesseract OCR on a preprocessed image.

Features:
  • Dynamic language selection  —  "hin+san" for Devanagari, "ben" for Bangla
  • Multi-PSM confidence extraction via image_to_data()
  • Character-level confidence aggregation
  • Graceful fallback: if the primary PSM fails, tries PSM 6 (assume uniform block)

Returns an OCRResult dataclass with raw text, per-word confidences, and
aggregate statistics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image


# ── language mapping ──────────────────────────────────────────────────────────

SCRIPT_TO_LANG = {
    "devanagari": "hin+san",
    "bangla":     "ben",
    "unknown":    "hin+san+ben",
}

# PSM modes to try in order (PSM 3 = auto, PSM 6 = uniform block)
_PSM_ORDER = [3, 6, 4]


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    text:           str   = ""
    language:       str   = ""
    psm_used:       int   = 3
    mean_conf:      float = 0.0   # average word confidence [0, 1]
    char_count:     int   = 0
    word_count:     int   = 0
    warnings:       List[str] = field(default_factory=list)

    # Per-word detail for optional display
    words:          List[str]  = field(default_factory=list)
    word_confs:     List[float] = field(default_factory=list)


# ── public API ────────────────────────────────────────────────────────────────

def run_ocr(
    image_np: np.ndarray,          # BGR uint8 numpy array (from PreprocessResult.final_np)
    script:   str = "unknown",
) -> OCRResult:
    """
    Run Tesseract on *image_np* and return an OCRResult.

    Args:
        image_np : BGR numpy array (output of preprocessing pipeline)
        script   : one of "devanagari", "bangla", "unknown"
    """
    lang = SCRIPT_TO_LANG.get(script, SCRIPT_TO_LANG["unknown"])
    gray = _ensure_gray(image_np)
    pil  = Image.fromarray(gray)

    result = OCRResult(language=lang)

    for psm in _PSM_ORDER:
        try:
            raw_text, words, confs = _tesseract_run(pil, lang, psm)
            if raw_text.strip():
                result.text       = raw_text
                result.psm_used   = psm
                result.words      = words
                result.word_confs = confs
                result.mean_conf  = _mean_conf(confs)
                result.char_count = len(raw_text.replace(" ", "").replace("\n", ""))
                result.word_count = len(words)
                return result
        except Exception as exc:
            result.warnings.append(f"PSM {psm} failed: {exc}")
            continue

    # All PSMs failed — return empty result
    result.warnings.append("All Tesseract PSM modes failed; returning empty text.")
    return result


# ── internal helpers ──────────────────────────────────────────────────────────

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _tesseract_run(
    pil_img: Image.Image,
    lang:    str,
    psm:     int,
) -> Tuple[str, List[str], List[float]]:
    """
    Returns (full_text, words, normalised_confidences[0..1]).
    Raises on tesseract error.
    """
    config = f"--oem 3 --psm {psm}"

    # Full text string
    full_text = pytesseract.image_to_string(pil_img, lang=lang, config=config)

    # Per-word detail
    data = pytesseract.image_to_data(
        pil_img, lang=lang, config=config,
        output_type=pytesseract.Output.DICT,
    )

    words:  List[str]  = []
    confs:  List[float] = []

    for word, conf in zip(data["text"], data["conf"]):
        word = str(word).strip()
        try:
            conf_val = int(conf)
        except (ValueError, TypeError):
            continue
        if conf_val < 0 or not word:   # -1 means block/line separator
            continue
        words.append(word)
        confs.append(round(conf_val / 100.0, 4))

    return full_text, words, confs


def _mean_conf(confs: List[float]) -> float:
    if not confs:
        return 0.0
    return round(sum(confs) / len(confs), 4)
