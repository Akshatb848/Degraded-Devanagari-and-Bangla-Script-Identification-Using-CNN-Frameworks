"""
Script Classifier
─────────────────
Identifies whether an image contains Devanagari or Bangla script.

Strategy (in priority order):
  1. CNN model  — loaded from models/script_classifier.h5 if it exists
  2. Unicode heuristic — counts Devanagari / Bangla codepoints in any
                         OCR text already extracted upstream
  3. Fallback — returns ("unknown", 0.0)

The heavy TensorFlow import is done lazily so the module can be imported
even when TensorFlow is not installed (Streamlit Cloud deployment).

Usage:
    from core.classifier import classify_script
    label, confidence = classify_script(image_bytes)
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / "models" / "script_classifier.h5"
INPUT_SIZE  = (64, 64)           # expected by the trained CNN

LABELS = {0: "devanagari", 1: "bangla"}

# Unicode ranges for heuristic fallback
_DEVANAGARI_RANGE = range(0x0900, 0x097F + 1)
_BANGLA_RANGE     = range(0x0980, 0x09FF + 1)


# ── Streamlit-cached loader ────────────────────────────────────────────────────

def get_classifier():
    """
    Return a loaded Keras model or None.

    Decorated with @st.cache_resource at call-site (streamlit_app.py) so that
    the model is loaded once per server process.  This function itself does NOT
    import streamlit so it remains usable outside a Streamlit context.
    """
    if not MODEL_PATH.exists():
        return None

    try:
        # Lazy import — TF not available on all deployments
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf          # noqa: F401 (triggers lazy C init)
        from tensorflow import keras     # noqa: F401

        model = keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception:
        return None


# ── public API ────────────────────────────────────────────────────────────────

def classify_script(
    image_bytes: bytes,
    model=None,          # pre-loaded Keras model (or None → heuristic only)
    ocr_text: str = "",  # optional: upstream OCR text for Unicode heuristic
) -> Tuple[str, float]:
    """
    Returns (script_label, confidence) where:
      script_label  in {"devanagari", "bangla", "unknown"}
      confidence    in [0.0, 1.0]
    """
    # 1 ── CNN ─────────────────────────────────────────────────────────────────
    if model is not None:
        try:
            label, conf = _cnn_predict(image_bytes, model)
            if label != "unknown":
                return label, conf
        except Exception:
            pass

    # 2 ── Unicode heuristic ───────────────────────────────────────────────────
    if ocr_text:
        label, conf = _unicode_heuristic(ocr_text)
        if label != "unknown":
            return label, conf

    # 3 ── pixel density heuristic (no text needed) ────────────────────────────
    label, conf = _pixel_heuristic(image_bytes)
    return label, conf


# ── CNN prediction ─────────────────────────────────────────────────────────────

def _cnn_predict(image_bytes: bytes, model) -> Tuple[str, float]:
    arr  = np.frombuffer(image_bytes, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "unknown", 0.0

    resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    tensor  = resized.astype(np.float32) / 255.0
    tensor  = tensor[np.newaxis, :, :, np.newaxis]   # (1, 64, 64, 1)

    probs = model.predict(tensor, verbose=0)[0]       # shape (num_classes,)

    if len(probs) == 1:
        # Binary sigmoid output
        p_bangla = float(probs[0])
        if p_bangla >= 0.5:
            return "bangla", round(p_bangla, 4)
        else:
            return "devanagari", round(1.0 - p_bangla, 4)

    # Softmax output [p_devanagari, p_bangla]
    idx  = int(np.argmax(probs))
    conf = round(float(probs[idx]), 4)
    return LABELS.get(idx, "unknown"), conf


# ── Unicode heuristic ─────────────────────────────────────────────────────────

def _unicode_heuristic(text: str) -> Tuple[str, float]:
    dev_count = sum(1 for ch in text if ord(ch) in _DEVANAGARI_RANGE)
    ban_count = sum(1 for ch in text if ord(ch) in _BANGLA_RANGE)
    total     = dev_count + ban_count

    if total == 0:
        return "unknown", 0.0

    if dev_count >= ban_count:
        return "devanagari", round(dev_count / total, 4)
    else:
        return "bangla", round(ban_count / total, 4)


# ── pixel density heuristic ───────────────────────────────────────────────────

def _pixel_heuristic(image_bytes: bytes) -> Tuple[str, float]:
    """
    Last-resort: return 'unknown' with 0.0 confidence.
    (A real pixel-level heuristic would need labelled feature statistics.)
    """
    return "unknown", 0.0
