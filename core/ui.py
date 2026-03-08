"""
UI Components
─────────────
Reusable Streamlit widgets for the AIOCR application.

All functions accept plain Python values and render directly to the
Streamlit page.  No global state is mutated here — state lives in
streamlit_app.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image


# ── pipeline progress ─────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "Preprocessing",
    "Script Classification",
    "OCR Extraction",
    "LLM Correction",
    "Validation",
]


def show_pipeline_progress(current_step: int) -> None:
    """
    Render a visual progress indicator for the 5-step pipeline.
    *current_step* is 0-indexed; pass len(PIPELINE_STEPS) when done.
    """
    n = len(PIPELINE_STEPS)
    cols = st.columns(n)
    for i, (col, label) in enumerate(zip(cols, PIPELINE_STEPS)):
        with col:
            if i < current_step:
                st.markdown(f"✅ **{label}**")
            elif i == current_step:
                st.markdown(f"⏳ **{label}**")
            else:
                st.markdown(f"⬜ {label}")


# ── confidence meter ──────────────────────────────────────────────────────────

def confidence_meter(
    label:      str,
    value:      float,          # [0.0, 1.0]
    warn_below: float = 0.5,
) -> None:
    """Render a labelled progress bar with colour-coded warning."""
    pct = round(value * 100, 1)
    color = "normal" if value >= warn_below else "inverse"
    st.metric(label=label, value=f"{pct}%", delta=None)
    st.progress(min(max(value, 0.0), 1.0))
    if value < warn_below:
        st.warning(f"Low confidence ({pct}%) — result may be unreliable.")


# ── before / after image viewer ───────────────────────────────────────────────

def before_after_images(
    original:   Image.Image,
    processed:  Image.Image,
    orig_label: str = "Original",
    proc_label: str = "Preprocessed",
) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{orig_label}**")
        st.image(original, use_container_width=True)
    with col2:
        st.markdown(f"**{proc_label}**")
        st.image(processed, use_container_width=True)


# ── preprocessing stage gallery ──────────────────────────────────────────────

def preprocessing_gallery(stages: Dict[str, Optional[Image.Image]]) -> None:
    """
    Render a 3-column grid of intermediate preprocessing images.

    Args:
        stages: ordered dict of {label: PIL Image or None}
    """
    items = [(k, v) for k, v in stages.items() if v is not None]
    if not items:
        return

    cols_per_row = 3
    rows = [items[i:i + cols_per_row] for i in range(0, len(items), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, (label, img) in zip(cols, row):
            with col:
                st.markdown(f"**{label}**")
                st.image(img, use_container_width=True)


# ── OCR / LLM text comparison ─────────────────────────────────────────────────

def text_comparison(
    raw_text:       str,
    corrected_text: str,
    corrections:    List[str],
    reasoning:      str,
) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw OCR Output**")
        st.text_area(
            label="raw_ocr",
            value=raw_text or "(no text extracted)",
            height=250,
            label_visibility="collapsed",
            disabled=True,
        )
    with col2:
        st.markdown("**LLM Corrected Output**")
        st.text_area(
            label="corrected",
            value=corrected_text or "(no correction)",
            height=250,
            label_visibility="collapsed",
            disabled=True,
        )

    if corrections:
        with st.expander("Corrections applied"):
            for i, c in enumerate(corrections, 1):
                st.markdown(f"{i}. {c}")

    if reasoning:
        with st.expander("Model reasoning"):
            st.write(reasoning)


# ── agent timing chart ────────────────────────────────────────────────────────

def agent_timing_chart(timings: Dict[str, float]) -> None:
    """
    Render a horizontal bar chart of per-step durations (ms).

    Args:
        timings: {step_name: duration_ms}
    """
    if not timings:
        return

    try:
        import pandas as pd

        df = pd.DataFrame(
            {"Step": list(timings.keys()), "Duration (ms)": list(timings.values())}
        ).set_index("Step")

        st.markdown("**Processing Time per Step**")
        st.bar_chart(df)
    except Exception:
        st.markdown("**Processing Time per Step**")
        for step, ms in timings.items():
            st.text(f"  {step}: {ms:.1f} ms")


# ── metrics row ───────────────────────────────────────────────────────────────

def metrics_row(
    script:           str,
    script_conf:      float,
    ocr_conf:         float,
    total_ms:         float,
    line_count:       int,
    char_count:       int,
) -> None:
    cols = st.columns(6)
    data = [
        ("Script",       script.title()),
        ("Script Conf.", f"{script_conf*100:.1f}%"),
        ("OCR Conf.",    f"{ocr_conf*100:.1f}%"),
        ("Time",         f"{total_ms:.0f} ms"),
        ("Lines",        str(line_count)),
        ("Chars",        str(char_count)),
    ]
    for col, (label, value) in zip(cols, data):
        col.metric(label, value)


# ── download button ───────────────────────────────────────────────────────────

def download_result_json(result_dict: dict, filename: str = "ocr_result.json") -> None:
    import json

    json_bytes = json.dumps(result_dict, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label="Download result (JSON)",
        data=json_bytes,
        file_name=filename,
        mime="application/json",
    )
