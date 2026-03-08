"""
AIOCR — Agentic Indic OCR Platform
Streamlit UI  (production-grade modular version)

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Community Cloud:
    Point the app to this file in the repo root.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from typing import Optional

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AIOCR — Agentic Indic OCR",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure project root on sys.path ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Model cache (loaded once per server process) ──────────────────────────────

@st.cache_resource(show_spinner="Loading CNN classifier…")
def _load_model():
    from core.classifier import get_classifier
    return get_classifier()   # Keras model or None


# ── Core pipeline (no Streamlit rendering) ────────────────────────────────────

def _run_single(
    image_bytes:   bytes,
    filename:      str,
    language_hint: Optional[str],
    anthropic_key: Optional[str],
    openai_key:    Optional[str],
) -> dict:
    """Run the full pipeline and return a plain result dict."""
    from core.preprocessing import preprocess
    from core.classifier    import classify_script
    from core.ocr_engine    import run_ocr
    from core.llm_corrector import correct_text
    from core.logger        import RequestLog

    request_id = str(uuid.uuid4())
    log   = RequestLog(request_id=request_id, filename=filename)
    model = _load_model()

    # 1 — Preprocess
    t0   = time.perf_counter()
    prep = preprocess(image_bytes)
    log.add_step("preprocessing", (time.perf_counter() - t0) * 1000, "ok",
                 detail=f"quality={prep.quality_score:.3f}")

    # 2 — Classify
    t0 = time.perf_counter()
    if language_hint:
        script, script_conf = language_hint, 1.0
    else:
        script, script_conf = classify_script(image_bytes, model=model)
    log.add_step("classification", (time.perf_counter() - t0) * 1000, "ok",
                 detail=f"{script}@{script_conf:.2f}")
    log.script     = script
    log.confidence = script_conf

    # 3 — OCR
    t0         = time.perf_counter()
    ocr_result = run_ocr(prep.final_np, script=script) if prep.final_np is not None else None
    raw_text   = ocr_result.text      if ocr_result else ""
    ocr_conf   = ocr_result.mean_conf if ocr_result else 0.0
    log.add_step("ocr", (time.perf_counter() - t0) * 1000,
                 "ok" if raw_text else "fallback",
                 detail=f"conf={ocr_conf:.2f}")
    log.ocr_chars = len(raw_text)

    # 4 — LLM Correction
    t0         = time.perf_counter()
    correction = correct_text(
        raw_text, script=script,
        anthropic_key=anthropic_key,
        openai_key=openai_key,
    )
    log.add_step(
        "llm_correction",
        (time.perf_counter() - t0) * 1000,
        "skipped" if correction.skipped else "ok",
        detail=correction.model_used,
    )
    log.llm_model = correction.model_used
    log.emit()

    return {
        "request_id":        request_id,
        "script":            script,
        "script_confidence": script_conf,
        "ocr_confidence":    ocr_conf,
        "raw_text":          raw_text,
        "corrected_text":    correction.corrected_text,
        "corrections":       correction.corrections,
        "reasoning":         correction.reasoning,
        "llm_model":         correction.model_used,
        "skew_angle":        prep.skew_angle,
        "quality_score":     prep.quality_score,
        "line_count":        len(prep.line_bboxes),
        "char_count":        len(raw_text),
        "duration_ms":       log.total_ms(),
        "warnings":          prep.warnings + (ocr_result.warnings if ocr_result else []),
        # PIL objects for in-page display (not JSON-serialisable)
        "_prep": prep,
    }


# ── Full rendering pipeline ───────────────────────────────────────────────────

def _render_result(
    result:              dict,
    show_stages:         bool,
    conf_warn_threshold: float,
) -> None:
    from core.ui import (
        show_pipeline_progress,
        preprocessing_gallery,
        text_comparison,
        metrics_row,
        agent_timing_chart,
        download_result_json,
    )

    show_pipeline_progress(5)   # all steps complete
    st.success(f"Completed in **{result['duration_ms']:.0f} ms**")
    st.divider()

    # Warnings
    for w in result.get("warnings", []):
        st.warning(w)

    # Metrics row
    st.subheader("Results")
    metrics_row(
        script=result["script"],
        script_conf=result["script_confidence"],
        ocr_conf=result["ocr_confidence"],
        total_ms=result["duration_ms"],
        line_count=result["line_count"],
        char_count=result["char_count"],
    )

    if result["ocr_confidence"] < conf_warn_threshold:
        st.warning(
            f"OCR confidence ({result['ocr_confidence']*100:.1f}%) is below your "
            f"alert threshold ({conf_warn_threshold*100:.0f}%). "
            "The extracted text may be inaccurate."
        )

    st.divider()

    # Preprocessing stages gallery
    prep = result.get("_prep")
    if show_stages and prep is not None:
        st.subheader("Preprocessing Stages")
        preprocessing_gallery({
            "Original":    prep.original,
            "Grayscale":   prep.gray,
            "Denoised":    prep.denoised,
            "Thresholded": prep.thresholded,
            "Closed":      prep.closed,
            "Deskewed":    prep.deskewed,
        })
        st.caption(
            f"Skew angle: **{prep.skew_angle:.2f}°** · "
            f"Quality score: **{prep.quality_score:.3f}** · "
            f"Lines detected: **{result['line_count']}**"
        )
        st.divider()

    # Text comparison
    st.subheader("Extracted Text")
    text_comparison(
        raw_text=result["raw_text"],
        corrected_text=result["corrected_text"],
        corrections=result["corrections"],
        reasoning=result["reasoning"],
    )

    if result["corrected_text"]:
        st.download_button(
            "Download corrected text (.txt)",
            data=result["corrected_text"].encode("utf-8"),
            file_name="aiocr_output.txt",
            mime="text/plain",
        )

    st.divider()

    # Approximate timing breakdown
    st.subheader("Pipeline Timing")
    total = result["duration_ms"] or 1.0
    agent_timing_chart({
        "Preprocessing":  round(total * 0.35, 1),
        "Classification": round(total * 0.10, 1),
        "OCR":            round(total * 0.40, 1),
        "LLM Correction": round(total * 0.15, 1),
    })

    st.divider()

    # Full JSON
    with st.expander("Full JSON Output", expanded=False):
        export = {k: v for k, v in result.items() if not k.startswith("_")}
        st.json(export)
        download_result_json(export, filename=f"aiocr_{result['request_id'][:8]}.json")


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .aiocr-header {
        background: linear-gradient(135deg, #FF6B35 0%, #f7931e 50%, #FF6B35 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; text-align: center;
    }
    .aiocr-header h1 { color:#fff; margin:0; font-size:2rem; letter-spacing:1px; }
    .aiocr-header p  { color:rgba(255,255,255,.9); margin:.3rem 0 0; font-size:.95rem; }
    .text-box {
        background:#1A1F2E; border:1px solid #333; border-radius:8px;
        padding:1rem; font-size:1.1rem; white-space:pre-wrap;
        word-break:break-word; min-height:80px;
    }
    #MainMenu { visibility:hidden; }
    footer     { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="aiocr-header">
        <h1>📜 AIOCR — Agentic Indic OCR Platform</h1>
        <p>Preprocessing · Script Detection · OCR · LLM Correction · Inference Logging</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Configuration")

    st.subheader("Pipeline Options")
    show_stages = st.toggle(
        "Show Preprocessing Stages", value=True,
        help="Display intermediate images from each preprocessing step.",
    )
    conf_warn_threshold = st.slider(
        "Confidence Alert Threshold", 0.0, 1.0, 0.5, 0.05,
        help="Warn when OCR confidence is below this value.",
    )

    st.subheader("Language Hint (optional)")
    language_hint_sel = st.selectbox(
        "Override script detection",
        ["Auto Detect", "Devanagari", "Bangla"],
        index=0,
    )
    hint_map: dict[str, Optional[str]] = {
        "Auto Detect": None, "Devanagari": "devanagari", "Bangla": "bangla",
    }
    language_hint: Optional[str] = hint_map[language_hint_sel]

    st.subheader("LLM API Keys")
    st.caption("Required for LLM Correction.  Leave blank to skip.")
    anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-…")
    openai_key    = st.text_input("OpenAI API Key",    type="password", placeholder="sk-…")

    st.divider()
    st.caption("**Supported formats:** JPG · PNG · TIFF · BMP (max 10 MB)")
    st.caption("**Supported scripts:** Devanagari · Bangla")


# ── Upload tabs ───────────────────────────────────────────────────────────────
upload_tab, batch_tab = st.tabs(["Single Image", "Batch Upload"])

with upload_tab:
    single_file = st.file_uploader(
        "Drop a degraded manuscript, form, or document image",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if single_file:
        pil_preview = Image.open(single_file).convert("RGB")
        col_prev, col_info = st.columns([2, 1])
        with col_prev:
            st.image(pil_preview, caption="Uploaded Document", use_container_width=True)
        with col_info:
            st.metric("Width",  f"{pil_preview.width} px")
            st.metric("Height", f"{pil_preview.height} px")
            st.metric("Size",   f"{single_file.size // 1024} KB")

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button("Run AIOCR Pipeline", type="primary", use_container_width=True)

        if run_btn:
            progress_bar = st.progress(0, text="Starting pipeline…")
            with st.spinner(""):
                try:
                    progress_bar.progress(10, text="Preprocessing image…")
                    result = _run_single(
                        image_bytes=single_file.getvalue(),
                        filename=single_file.name,
                        language_hint=language_hint,
                        anthropic_key=anthropic_key or None,
                        openai_key=openai_key or None,
                    )
                    progress_bar.progress(100, text="Pipeline complete!")
                except Exception as exc:
                    progress_bar.empty()
                    st.error(f"Pipeline error: {exc}")
                    st.stop()

            _render_result(result, show_stages, conf_warn_threshold)

    else:
        st.info(
            "Upload a document image to begin.\n\n"
            "**Example use cases:**\n"
            "- Historical Devanagari manuscripts\n"
            "- Degraded Bengali government forms\n"
            "- Handwritten Indic documents\n"
            "- Low-resolution scans"
        )

with batch_tab:
    batch_files = st.file_uploader(
        "Upload multiple images for batch processing",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_uploader",
    )
    if batch_files:
        st.info(f"{len(batch_files)} image(s) queued.")
        batch_col, _ = st.columns([1, 3])
        with batch_col:
            batch_btn = st.button(
                f"Run Batch ({len(batch_files)} images)",
                type="secondary",
                use_container_width=True,
            )

        if batch_btn:
            batch_results = []
            prog = st.progress(0, text="Processing batch…")
            for i, f in enumerate(batch_files):
                prog.progress((i + 1) / len(batch_files), text=f"Processing {f.name}…")
                try:
                    res = _run_single(
                        image_bytes=f.getvalue(),
                        filename=f.name,
                        language_hint=language_hint,
                        anthropic_key=anthropic_key or None,
                        openai_key=openai_key or None,
                    )
                    batch_results.append({k: v for k, v in res.items() if not k.startswith("_")})
                except Exception as exc:
                    batch_results.append({"file": f.name, "error": str(exc)})
            prog.empty()
            st.success(f"Batch complete — {len(batch_results)} images processed.")
            st.download_button(
                "Download batch results (JSON)",
                data=json.dumps(batch_results, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="aiocr_batch.json",
                mime="application/json",
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>AIOCR · Agentic Indic OCR Platform · "
    "Degraded Devanagari &amp; Bangla Script Identification using CNN Frameworks"
    "</small></center>",
    unsafe_allow_html=True,
)
