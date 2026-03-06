"""
AIOCR — Agentic Indic OCR Platform
Streamlit UI

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Community Cloud:
    Point the app to this file in the repo root.
"""

import base64
import io
import os
import sys
import time
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

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Top header bar */
    .aiocr-header {
        background: linear-gradient(135deg, #FF6B35 0%, #f7931e 50%, #FF6B35 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .aiocr-header h1 { color: #fff; margin: 0; font-size: 2rem; letter-spacing: 1px; }
    .aiocr-header p  { color: rgba(255,255,255,0.9); margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* Agent timeline card */
    .agent-card {
        background: #1A1F2E;
        border-left: 4px solid #FF6B35;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
    }
    .agent-card.completed { border-color: #4CAF50; }
    .agent-card.skipped   { border-color: #9E9E9E; }
    .agent-card.failed    { border-color: #f44336; }

    /* Confidence pill */
    .conf-pill {
        display: inline-block;
        background: #FF6B35;
        color: #fff;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }

    /* Script badge */
    .script-badge {
        display: inline-block;
        background: #1A237E;
        color: #fff;
        border-radius: 6px;
        padding: 4px 14px;
        font-size: 1rem;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Result text box */
    .text-box {
        background: #1A1F2E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.1rem;
        white-space: pre-wrap;
        word-break: break-word;
        min-height: 80px;
    }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="aiocr-header">
        <h1>📜 AIOCR — Agentic Indic OCR Platform</h1>
        <p>7-Agent AI Pipeline · Script Detection · Image Restoration · OCR · LLM Correction · RAG</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — Configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Devanagari_a.svg/120px-Devanagari_a.svg.png",
        width=60,
    )
    st.title("Configuration")

    st.subheader("Pipeline Options")
    enable_restoration = st.toggle("Image Restoration", value=True,
                                   help="Apply denoising, deskew, CLAHE, and bicubic upscaling.")
    enable_rag = st.toggle("Knowledge Retrieval (RAG)", value=False,
                           help="Requires a local ChromaDB instance. Disable for Streamlit Cloud.")
    show_annotated = st.toggle("Show Annotated Image", value=True,
                               help="Draw bounding boxes on the detected text regions.")

    st.subheader("Language Hint (optional)")
    language_hint = st.selectbox(
        "Override script detection",
        ["Auto Detect", "Devanagari", "Bangla"],
        index=0,
    )
    hint_map = {"Auto Detect": None, "Devanagari": "devanagari", "Bangla": "bangla"}
    language_hint_val: Optional[str] = hint_map[language_hint]

    st.subheader("LLM API Keys")
    st.caption("Needed for Agent 5 (LLM Correction). Leave blank to skip correction.")
    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Used for Claude-based OCR correction.",
    )
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Fallback if Anthropic key is not set.",
    )

    st.divider()
    st.caption("**Supported formats:** JPG, PNG, TIFF, BMP (max 10 MB)")
    st.caption("**Supported scripts:** Devanagari · Bangla")


# ── Main Upload Area ──────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Drop a degraded manuscript, form, or document image",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Uploaded Document", use_container_width=True)
        st.caption(f"Size: {pil_img.width}×{pil_img.height} px · {uploaded_file.size // 1024} KB")

with col_preview:
    if not uploaded_file:
        st.info(
            "Upload a document image to begin.\n\n"
            "**Example use cases:**\n"
            "- Historical Devanagari manuscripts\n"
            "- Degraded Bengali government forms\n"
            "- Handwritten Indic documents\n"
            "- Low-resolution scans"
        )

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if uploaded_file:
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("Run AIOCR Pipeline", type="primary", use_container_width=True)

    if run_btn:
        image_bytes = uploaded_file.getvalue()

        # Progress tracking
        progress_bar = st.progress(0, text="Starting pipeline…")
        status_text = st.empty()

        def progress_callback(step: int, total: int, label: str):
            pct = int((step / total) * 100)
            progress_bar.progress(pct, text=label)
            status_text.caption(f"Step {step}/{total}: {label}")

        with st.spinner(""):
            try:
                from agents.pipeline_runner import run_full_pipeline

                start_time = time.time()
                result = run_full_pipeline(
                    image_bytes=image_bytes,
                    language_hint=language_hint_val,
                    enable_restoration=enable_restoration,
                    enable_rag=enable_rag,
                    anthropic_key=anthropic_key or None,
                    openai_key=openai_key or None,
                    progress_callback=progress_callback,
                )
                total_time = round((time.time() - start_time) * 1000, 1)
                progress_bar.progress(100, text="Pipeline complete!")
                status_text.empty()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Pipeline error: {e}")
                st.stop()

        st.success(f"Pipeline completed in **{total_time} ms**")
        st.divider()

        # ── Results: Row 1 — Script & Confidence ─────────────────────────────
        st.subheader("Results")
        m1, m2, m3, m4 = st.columns(4)

        script = result.get("script", "unknown").capitalize()
        conf = result.get("overall_confidence", 0.0)
        ocr_conf = result.get("ocr_confidence", 0.0)
        script_conf = result.get("script_confidence", 0.0)

        m1.metric("Detected Script", script)
        m2.metric("Script Confidence", f"{script_conf * 100:.1f}%")
        m3.metric("OCR Confidence", f"{ocr_conf * 100:.1f}%")
        m4.metric("Overall Confidence", f"{conf * 100:.1f}%")

        st.divider()

        # ── Results: Row 2 — Images ───────────────────────────────────────────
        if enable_restoration or show_annotated:
            img_cols = st.columns(2 if (enable_restoration and show_annotated) else 1)

            if enable_restoration and result.get("restored_image_bytes"):
                with img_cols[0]:
                    st.subheader("Restored Image")
                    rest_pil = Image.open(io.BytesIO(result["restored_image_bytes"]))
                    st.image(rest_pil, use_container_width=True)
                    enhancements = result.get("restoration", {}).get("enhancements", [])
                    q_before = result.get("restoration", {}).get("quality_before", 0)
                    q_after  = result.get("restoration", {}).get("quality_after", 0)
                    st.caption(f"Enhancements: `{'` · `'.join(enhancements) or 'none'}`")
                    st.caption(f"Quality: {q_before:.3f} → {q_after:.3f}")

            ann_col = img_cols[-1] if (enable_restoration and show_annotated) else img_cols[0]
            if show_annotated and result.get("annotated_image_bytes"):
                with ann_col:
                    st.subheader("Text Regions Detected")
                    ann_pil = Image.open(io.BytesIO(result["annotated_image_bytes"]))
                    st.image(ann_pil, use_container_width=True)
                    st.caption(
                        f"Detected **{len(result.get('regions', []))}** text regions "
                        f"via `{result.get('text_detection_method', '')}`"
                    )

            st.divider()

        # ── Results: Row 3 — OCR Text ─────────────────────────────────────────
        st.subheader("Extracted Text")
        t1, t2 = st.columns(2, gap="large")

        with t1:
            st.markdown("**Raw OCR Output**")
            raw = result.get("raw_text", "") or "_No text detected_"
            st.markdown(f'<div class="text-box">{raw}</div>', unsafe_allow_html=True)
            st.caption(f"OCR engine: `{result.get('ocr_method', 'none')}`")

        with t2:
            st.markdown("**LLM-Corrected Text**")
            corrected = result.get("final_text", "") or "_No text detected_"
            st.markdown(f'<div class="text-box">{corrected}</div>', unsafe_allow_html=True)
            llm_model = result.get("llm_model", "none")
            st.caption(f"LLM model: `{llm_model}`")

        # Copy button for corrected text
        if result.get("final_text"):
            st.download_button(
                "Download Extracted Text",
                data=result["final_text"].encode("utf-8"),
                file_name="aiocr_output.txt",
                mime="text/plain",
            )

        st.divider()

        # ── Results: Row 4 — Corrections & Reasoning ─────────────────────────
        corrections = result.get("corrections", [])
        reasoning   = result.get("reasoning", "")

        if corrections or reasoning:
            st.subheader("LLM Correction Details")
            c1, c2 = st.columns([1, 1], gap="large")

            with c1:
                st.markdown("**Corrections Made**")
                if corrections:
                    for c in corrections:
                        st.markdown(f"- {c}")
                else:
                    st.caption("No corrections were necessary.")

            with c2:
                st.markdown("**Reasoning**")
                st.info(reasoning or "—")

            st.divider()

        # ── Results: Row 5 — Agent Timeline ──────────────────────────────────
        st.subheader("Agent Processing Timeline")
        timings = result.get("agent_timings", {})

        AGENT_ORDER = [
            ("ScriptDetection",    "1️⃣  Script Detection Agent",      "CNN ensemble"),
            ("ImageRestoration",   "2️⃣  Image Restoration Agent",     "CLAHE + denoising + bicubic"),
            ("TextDetection",      "3️⃣  Text Detection Agent",        "Connected components"),
            ("CharRecognition",    "4️⃣  Character Recognition Agent", "Tesseract / TrOCR"),
            ("LLMCorrection",      "5️⃣  LLM Correction Agent",       "Claude / GPT-4o"),
            ("KnowledgeRetrieval", "6️⃣  Knowledge Retrieval Agent",  "ChromaDB RAG"),
            ("OutputFormatting",   "7️⃣  Output Formatting Agent",    "Bounding boxes + JSON"),
        ]

        # Bar chart of timings
        import pandas as pd
        timing_df = pd.DataFrame([
            {"Agent": label, "Time (ms)": timings.get(key, 0)}
            for key, label, _ in AGENT_ORDER
        ])
        st.bar_chart(timing_df.set_index("Agent"), color="#FF6B35", height=200)

        # Detail rows
        for key, label, description in AGENT_ORDER:
            ms = timings.get(key)
            if ms is not None:
                skipped = (key == "KnowledgeRetrieval" and not enable_rag) or \
                          (key == "ImageRestoration" and not enable_restoration)
                status_cls = "skipped" if skipped else "completed"
                badge = "⚪ skipped" if skipped else f"✅ {ms} ms"
                st.markdown(
                    f'<div class="agent-card {status_cls}">'
                    f"<b>{label}</b> &nbsp;·&nbsp; <small>{description}</small>"
                    f"<span style='float:right'>{badge}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Results: Row 6 — JSON Output ─────────────────────────────────────
        with st.expander("Full JSON Output", expanded=False):
            import json

            json_output = {
                "request_id": result.get("request_id"),
                "script": result.get("script"),
                "script_confidence": result.get("script_confidence"),
                "overall_confidence": result.get("overall_confidence"),
                "language": {
                    "devanagari": "Hindi / Sanskrit / Marathi",
                    "bangla": "Bengali",
                }.get(result.get("script", ""), "Unknown"),
                "raw_text": result.get("raw_text", ""),
                "corrected_text": result.get("final_text", ""),
                "corrections": result.get("corrections", []),
                "reasoning": result.get("reasoning", ""),
                "bounding_boxes": [
                    r["bbox"] for r in result.get("regions", [])
                ],
                "agent_timings_ms": result.get("agent_timings", {}),
                "enhancements_applied": result.get("restoration", {}).get("enhancements", []),
                "ocr_method": result.get("ocr_method", ""),
                "llm_model": result.get("llm_model", "none"),
            }

            st.json(json_output)
            st.download_button(
                "Download JSON",
                data=json.dumps(json_output, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="aiocr_result.json",
                mime="application/json",
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>AIOCR · Agentic Indic OCR Platform · "
    "Degraded Devanagari &amp; Bangla Script Identification using CNN Frameworks</small></center>",
    unsafe_allow_html=True,
)
