"""
Multi-Agent Manuscript Intelligence System
──────────────────────────────────────────
Production-grade Streamlit UI — dark-mode research platform.

4-agent sequential pipeline:
  Agent 1 — Image Restoration
  Agent 2 — Script Detection
  Agent 3 — OCR Extraction
  Agent 4 — Linguistic Reconstruction
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manuscript Intelligence System",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising CNN classifier…")
def _load_model():
    from core.classifier import get_classifier
    return get_classifier()


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — dark-mode research platform
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── global ── */
.stApp { background: #080C18; }
section[data-testid="stSidebar"] { background: #0D1120 !important; }
section[data-testid="stSidebar"] * { color: #E0E0E0; }

/* ── hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #0D1120 0%, #141a35 40%, #0D1120 100%);
    border: 1px solid rgba(255,107,53,.35);
    border-radius: 18px;
    padding: 3rem 2.5rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%,
        rgba(255,107,53,.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: .5px;
    background: linear-gradient(90deg, #FF6B35, #f7c59f, #FF6B35);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 0 0 .6rem;
}
.hero-sub {
    color: rgba(255,255,255,.65);
    font-size: 1rem;
    margin: 0;
    letter-spacing: .3px;
}
.hero-badges { margin-top: 1.2rem; }
.badge {
    display: inline-block;
    background: rgba(255,107,53,.15);
    border: 1px solid rgba(255,107,53,.4);
    color: #FF6B35;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .5px;
    margin: 0 4px;
    text-transform: uppercase;
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

/* ── section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FF6B35;
    letter-spacing: .4px;
    border-left: 3px solid #FF6B35;
    padding-left: .7rem;
    margin: 1.6rem 0 1rem;
}

/* ── upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,107,53,.05) !important;
    border: 2px dashed rgba(255,107,53,.4) !important;
    border-radius: 12px !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255,107,53,.8) !important;
}

/* ── pipeline ── */
.pipeline-outer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.6rem 1.2rem;
    background: #0D1120;
    border: 1px solid #1e2540;
    border-radius: 14px;
    margin: 1.2rem 0 1.8rem;
    gap: 0;
}
.agent-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: .5rem;
    min-width: 120px;
    position: relative;
    z-index: 1;
}
.agent-icon {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    border: 2px solid transparent;
    transition: all .3s ease;
}
/* pending */
.agent-node.pending  .agent-icon { background: #1a1f30; border-color: #2a3050; }
.agent-node.pending  .agent-label { color: #4a5070; }
/* active */
.agent-node.active   .agent-icon {
    background: rgba(255,107,53,.18);
    border-color: #FF6B35;
    box-shadow: 0 0 18px rgba(255,107,53,.5);
    animation: node-pulse 1.2s ease-in-out infinite;
}
.agent-node.active   .agent-label { color: #FF6B35; font-weight: 700; }
/* done */
.agent-node.done     .agent-icon { background: rgba(72,199,142,.15); border-color: #48c78e; }
.agent-node.done     .agent-label { color: #48c78e; font-weight: 600; }
/* error */
.agent-node.error    .agent-icon { background: rgba(255,92,92,.15); border-color: #ff5c5c; }
.agent-node.error    .agent-label { color: #ff5c5c; }

.agent-label { font-size: .78rem; text-align: center; letter-spacing: .3px; }
.agent-status-pill {
    font-size: .68rem;
    padding: 2px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,.06);
    color: #aaa;
    margin-top: 2px;
}
.agent-node.done  .agent-status-pill { background: rgba(72,199,142,.15); color: #48c78e; }
.agent-node.active .agent-status-pill { background: rgba(255,107,53,.15); color: #FF6B35; }

/* connector */
.pipe-connector {
    flex: 1;
    height: 3px;
    background: #1e2540;
    position: relative;
    margin: 0 6px;
    border-radius: 2px;
    overflow: hidden;
}
.pipe-connector .fill {
    height: 100%;
    background: linear-gradient(90deg, #FF6B35, #f7c59f);
    width: 0%;
    transition: width .6s ease;
    border-radius: 2px;
}
.pipe-connector.done .fill { width: 100%; }

@keyframes node-pulse {
    0%, 100% { box-shadow: 0 0 18px rgba(255,107,53,.5); }
    50%       { box-shadow: 0 0 32px rgba(255,107,53,.9); }
}

/* ── result cards ── */
.result-card {
    background: #0D1120;
    border: 1px solid #1e2540;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    animation: fadeSlideIn .4s ease forwards;
}
.result-card-title {
    font-size: .8rem;
    font-weight: 700;
    color: #FF6B35;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: .6rem;
}
.result-text {
    font-size: 1.05rem;
    color: #e8e8e8;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.7;
    min-height: 60px;
}
.result-text.mono { font-family: 'Courier New', monospace; font-size: .9rem; }

/* ── confidence gauge ── */
.conf-gauge-wrap { margin: .4rem 0 .8rem; }
.conf-gauge-label {
    display: flex;
    justify-content: space-between;
    font-size: .78rem;
    color: #aaa;
    margin-bottom: .3rem;
}
.conf-gauge-bar {
    height: 8px;
    background: #1e2540;
    border-radius: 4px;
    overflow: hidden;
}
.conf-gauge-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1.2s ease;
}
.conf-high   { background: linear-gradient(90deg, #48c78e, #06d6a0); }
.conf-medium { background: linear-gradient(90deg, #f7c59f, #FF6B35); }
.conf-low    { background: linear-gradient(90deg, #ff5c5c, #ff8c8c); }

/* ── metric chips ── */
.metric-row { display: flex; flex-wrap: wrap; gap: .8rem; margin: .8rem 0; }
.metric-chip {
    background: #131828;
    border: 1px solid #1e2540;
    border-radius: 10px;
    padding: .55rem 1.1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 90px;
}
.chip-label { font-size: .68rem; color: #666; text-transform: uppercase; letter-spacing: .5px; }
.chip-value { font-size: 1.15rem; font-weight: 700; color: #e0e0e0; margin-top: 2px; }
.chip-value.accent { color: #FF6B35; }

/* ── image viewer ── */
.img-card {
    background: #0D1120;
    border: 1px solid #1e2540;
    border-radius: 12px;
    padding: .8rem;
    text-align: center;
}
.img-card-label {
    font-size: .75rem;
    font-weight: 700;
    color: #666;
    text-transform: uppercase;
    letter-spacing: .6px;
    margin-bottom: .5rem;
}

/* ── scroll animation ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── hide streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML helpers
# ══════════════════════════════════════════════════════════════════════════════

def _agent_node(idx: int, icon: str, label: str, state: str, timing: str = "") -> str:
    pill = {"pending": "Waiting", "active": "Processing…",
            "done": timing or "Done", "error": "Error"}.get(state, "")
    return f"""
    <div class="agent-node {state}">
        <div class="agent-icon">{icon}</div>
        <div class="agent-label">{label}</div>
        <div class="agent-status-pill">{pill}</div>
    </div>
    """

def _connector(done: bool) -> str:
    cls = "done" if done else ""
    return f'<div class="pipe-connector {cls}"><div class="fill"></div></div>'

def _render_pipeline(states: list[str], timings: list[str]) -> str:
    icons  = ["🔧", "🔍", "📝", "🧠"]
    labels = ["Image<br>Restoration", "Script<br>Detection",
              "OCR<br>Extraction", "Linguistic<br>Reconstruction"]
    parts  = ['<div class="pipeline-outer">']
    for i, (icon, label, state, timing) in enumerate(zip(icons, labels, states, timings)):
        parts.append(_agent_node(i, icon, label, state, timing))
        if i < 3:
            parts.append(_connector(state == "done" and i < len(states)-1
                                    and states[i+1] in ("active","done")))
    parts.append("</div>")
    return "".join(parts)

def _conf_bar(label: str, value: float, level: str = "") -> str:
    if not level:
        level = "high" if value >= .75 else ("medium" if value >= .45 else "low")
    pct = round(value * 100, 1)
    return f"""
    <div class="conf-gauge-wrap">
        <div class="conf-gauge-label"><span>{label}</span><span>{pct}%</span></div>
        <div class="conf-gauge-bar">
            <div class="conf-gauge-fill conf-{level}" style="width:{pct}%"></div>
        </div>
    </div>"""

def _metric_chip(label: str, value: str, accent: bool = False) -> str:
    cls = "accent" if accent else ""
    return f"""<div class="metric-chip">
        <span class="chip-label">{label}</span>
        <span class="chip-value {cls}">{value}</span>
    </div>"""

def _result_card(title: str, content: str, mono: bool = False) -> str:
    mono_cls = "mono" if mono else ""
    return f"""
    <div class="result-card">
        <div class="result-card-title">{title}</div>
        <div class="result-text {mono_cls}">{content or '<em style="color:#555">—</em>'}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline execution
# ══════════════════════════════════════════════════════════════════════════════

def _execute_pipeline(
    image_bytes:         bytes,
    filename:            str,
    language_hint:       Optional[str],
    anthropic_key:       Optional[str],
    openai_key:          Optional[str],
    pipeline_slot:       "st.delta_generator.DeltaGenerator",
    progress_slot:       "st.delta_generator.DeltaGenerator",
) -> dict:
    """
    Run the 4-agent pipeline, updating *pipeline_slot* after each agent.
    Returns a result dict.
    """
    from core.preprocessing import preprocess
    from core.classifier    import classify_script
    from core.ocr_engine    import run_ocr
    from core.llm_corrector import correct_text
    from core.logger        import RequestLog

    N = 4
    states  = ["pending"] * N
    timings = [""] * N
    model   = _load_model()
    log     = RequestLog(request_id=str(uuid.uuid4()), filename=filename)

    def _draw(active: int):
        pipeline_slot.markdown(_render_pipeline(states, timings), unsafe_allow_html=True)
        progress_slot.progress(int(active / N * 100),
                               text=f"Agent {active}/{N} complete")

    # ── defaults ──────────────────────────────────────────────────────────────
    result: dict = {
        "request_id": log.request_id, "script": "unknown",
        "script_confidence": 0.0, "ocr_confidence": 0.0,
        "raw_text": "", "corrected_text": "",
        "reasoning": "", "confidence_label": "", "confidence_score": 0.0,
        "llm_model": "none", "restoration_summary": "",
        "skew_angle": 0.0, "quality_score": 0.0,
        "line_count": 0, "char_count": 0, "duration_ms": 0.0,
        "warnings": [], "_prep": None,
    }

    # ────────────────────────────────────────────────────────────────────────
    # Agent 1 — Image Restoration
    # ────────────────────────────────────────────────────────────────────────
    states[0] = "active"; _draw(0)
    t0 = time.perf_counter()
    try:
        prep = preprocess(image_bytes)
        dur = round((time.perf_counter() - t0) * 1000, 1)
        log.add_step("preprocessing", dur, "ok",
                     detail=f"quality={prep.quality_score:.3f}, skew={prep.skew_angle:.1f}°")
        result.update({
            "skew_angle":           prep.skew_angle,
            "quality_score":        prep.quality_score,
            "line_count":           len(prep.line_bboxes),
            "restoration_summary":  (
                f"Denoised · Thresholded · Deskewed ({prep.skew_angle:.1f}°) · "
                f"{len(prep.line_bboxes)} lines · quality {prep.quality_score:.3f}"
            ),
            "warnings":  prep.warnings,
            "_prep":     prep,
        })
        states[0] = "done"; timings[0] = f"{dur:.0f} ms"
    except Exception as exc:
        states[0] = "error"; timings[0] = "failed"
        result["warnings"].append(f"Restoration failed: {exc}")
        prep = None
    _draw(1)

    # ────────────────────────────────────────────────────────────────────────
    # Agent 2 — Script Detection
    # ────────────────────────────────────────────────────────────────────────
    states[1] = "active"; _draw(1)
    t0 = time.perf_counter()
    try:
        if language_hint:
            script, script_conf = language_hint, 1.0
        else:
            script, script_conf = classify_script(image_bytes, model=model)
        dur = round((time.perf_counter() - t0) * 1000, 1)
        log.add_step("classification", dur, "ok",
                     detail=f"{script}@{script_conf:.2f}")
        log.script = script; log.confidence = script_conf
        result["script"] = script
        result["script_confidence"] = script_conf
        states[1] = "done"; timings[1] = f"{dur:.0f} ms"
    except Exception as exc:
        states[1] = "error"; timings[1] = "failed"
        result["warnings"].append(f"Script detection failed: {exc}")
        script, script_conf = "unknown", 0.0
    _draw(2)

    # ────────────────────────────────────────────────────────────────────────
    # Agent 3 — OCR Extraction
    # ────────────────────────────────────────────────────────────────────────
    states[2] = "active"; _draw(2)
    t0 = time.perf_counter()
    try:
        final_np   = prep.final_np if prep else None
        ocr_result = run_ocr(final_np, script=script) if final_np is not None else None
        raw_text   = ocr_result.text      if ocr_result else ""
        ocr_conf   = ocr_result.mean_conf if ocr_result else 0.0
        dur        = round((time.perf_counter() - t0) * 1000, 1)
        log.add_step("ocr", dur, "ok" if raw_text else "fallback",
                     detail=f"conf={ocr_conf:.2f}, chars={len(raw_text)}")
        log.ocr_chars = len(raw_text)
        result["raw_text"]      = raw_text
        result["ocr_confidence"] = ocr_conf
        result["char_count"]    = len(raw_text)
        if ocr_result:
            result["warnings"] += ocr_result.warnings
        states[2] = "done"; timings[2] = f"{dur:.0f} ms"
    except Exception as exc:
        states[2] = "error"; timings[2] = "failed"
        result["warnings"].append(f"OCR failed: {exc}")
        raw_text = ""
    _draw(3)

    # ────────────────────────────────────────────────────────────────────────
    # Agent 4 — Linguistic Reconstruction
    # ────────────────────────────────────────────────────────────────────────
    states[3] = "active"; _draw(3)
    t0 = time.perf_counter()
    try:
        correction = correct_text(
            raw_text, script=script,
            anthropic_key=anthropic_key,
            openai_key=openai_key,
        )
        dur = round((time.perf_counter() - t0) * 1000, 1)
        log.add_step("llm_correction", dur,
                     "skipped" if correction.skipped else "ok",
                     detail=correction.model_used)
        log.llm_model = correction.model_used
        result.update({
            "corrected_text":    correction.corrected_text,
            "reasoning":         correction.reasoning,
            "confidence_label":  correction.confidence_label,
            "confidence_score":  correction.confidence_score,
            "llm_model":         correction.model_used,
        })
        states[3] = "done"; timings[3] = f"{dur:.0f} ms"
    except Exception as exc:
        states[3] = "error"; timings[3] = "failed"
        result["corrected_text"] = raw_text
        result["warnings"].append(f"Reconstruction failed: {exc}")
    _draw(N)

    log.emit()
    result["duration_ms"] = log.total_ms()
    progress_slot.progress(100, text="Pipeline complete ✓")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Results renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render_results(result: dict, original_bytes: bytes) -> None:
    st.markdown('<div class="section-header">RESULTS DASHBOARD</div>',
                unsafe_allow_html=True)

    # ── Row 1: image comparison + metrics ────────────────────────────────────
    img_col, metrics_col = st.columns([1.4, 1], gap="large")

    with img_col:
        prep = result.get("_prep")
        if prep is not None:
            l, r = st.columns(2)
            with l:
                st.markdown('<div class="img-card"><div class="img-card-label">Original</div>',
                            unsafe_allow_html=True)
                st.image(Image.open(io.BytesIO(original_bytes)).convert("RGB"),
                         use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with r:
                st.markdown('<div class="img-card"><div class="img-card-label">Restored</div>',
                            unsafe_allow_html=True)
                st.image(prep.deskewed, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.image(Image.open(io.BytesIO(original_bytes)).convert("RGB"),
                     use_container_width=True)

    with metrics_col:
        st.markdown('<div class="section-header">DETECTION METRICS</div>',
                    unsafe_allow_html=True)

        script_disp  = result["script"].capitalize()
        lang_map     = {"devanagari": "Sanskrit / Hindi", "bangla": "Bangla", "unknown": "Unknown"}
        lang_disp    = lang_map.get(result["script"], "Unknown")
        sc           = result["script_confidence"]
        oc           = result["ocr_confidence"]
        rc           = result["confidence_score"]
        conf_lbl     = result.get("confidence_label", "")

        chips = (
            _metric_chip("Detected Script", script_disp, True)
            + _metric_chip("Language", lang_disp)
            + _metric_chip("Lines", str(result["line_count"]))
            + _metric_chip("Characters", str(result["char_count"]))
            + _metric_chip("Total Time", f"{result['duration_ms']:.0f} ms")
        )
        st.markdown(f'<div class="metric-row">{chips}</div>', unsafe_allow_html=True)

        conf_bars = (
            _conf_bar("Script Detection Confidence", sc)
            + _conf_bar("OCR Quality Estimate", oc)
            + (f"<br>{_conf_bar(f'Reconstruction Confidence ({conf_lbl})', rc)}"
               if conf_lbl else "")
        )
        st.markdown(conf_bars, unsafe_allow_html=True)

        # Quality / skew
        if result["quality_score"] > 0:
            st.markdown(
                f'<small style="color:#555">Image quality score: '
                f'<b style="color:#888">{result["quality_score"]:.3f}</b> · '
                f'Skew corrected: <b style="color:#888">{result["skew_angle"]:.1f}°</b></small>',
                unsafe_allow_html=True,
            )

        # LLM confidence badge
        if conf_lbl:
            badge = {"High": ("🟢", "#48c78e"), "Medium": ("🟡", "#f7c59f"),
                     "Low":  ("🔴", "#ff5c5c")}.get(conf_lbl, ("⚪", "#888"))
            st.markdown(
                f'<div style="margin-top:.8rem;padding:.6rem 1rem;'
                f'background:rgba(255,255,255,.04);border-radius:8px;'
                f'border-left:3px solid {badge[1]};font-size:.85rem;color:{badge[1]}">'
                f'{badge[0]} &nbsp;Reconstruction Confidence: <b>{conf_lbl}</b></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Row 2: restoration summary ────────────────────────────────────────────
    if result["restoration_summary"]:
        st.markdown(
            _result_card("🔧 Image Restoration Summary", result["restoration_summary"]),
            unsafe_allow_html=True,
        )

    # ── Row 3: preprocessing stages ───────────────────────────────────────────
    prep = result.get("_prep")
    if prep is not None:
        with st.expander("Preprocessing Stage Gallery", expanded=False):
            stages = [
                ("Grayscale",    prep.gray),
                ("Denoised",     prep.denoised),
                ("Thresholded",  prep.thresholded),
                ("Closed",       prep.closed),
                ("Deskewed",     prep.deskewed),
            ]
            cols = st.columns(len(stages))
            for col, (label, img) in zip(cols, stages):
                with col:
                    st.markdown(f'<small style="color:#666;font-weight:600">{label}</small>',
                                unsafe_allow_html=True)
                    st.image(img, use_container_width=True)

    st.divider()

    # ── Row 4: text output ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">EXTRACTED &amp; RECONSTRUCTED TEXT</div>',
                unsafe_allow_html=True)

    tc1, tc2 = st.columns(2, gap="large")
    with tc1:
        st.markdown(
            _result_card("📝 Raw OCR Output (Agent 3)", result["raw_text"], mono=True),
            unsafe_allow_html=True,
        )
    with tc2:
        st.markdown(
            _result_card("🧠 Reconstructed Manuscript Text (Agent 4)",
                         result["corrected_text"]),
            unsafe_allow_html=True,
        )

    if result["reasoning"]:
        with st.expander("Linguistic Reconstruction Reasoning"):
            st.markdown(
                f'<div style="color:#c8c8c8;font-size:.9rem;line-height:1.7">'
                f'{result["reasoning"]}</div>',
                unsafe_allow_html=True,
            )

    # download buttons
    dc1, dc2, dc3 = st.columns(3)
    if result["raw_text"]:
        dc1.download_button("Download Raw OCR (.txt)",
                            data=result["raw_text"].encode("utf-8"),
                            file_name="raw_ocr.txt", mime="text/plain")
    if result["corrected_text"]:
        dc2.download_button("Download Reconstructed (.txt)",
                            data=result["corrected_text"].encode("utf-8"),
                            file_name="reconstructed.txt", mime="text/plain")
    export = {k: v for k, v in result.items() if not k.startswith("_")}
    dc3.download_button(
        "Download Full Report (.json)",
        data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"manuscript_{result['request_id'][:8]}.json",
        mime="application/json",
    )

    # ── Row 5: warnings ────────────────────────────────────────────────────────
    for w in result.get("warnings", []):
        st.warning(w)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="hero-title">Multi-Agent Manuscript Intelligence System</p>
    <p class="hero-sub">
        A 4-agent AI pipeline that restores, detects, extracts, and reconstructs
        degraded Indic manuscript text with linguistic precision.
    </p>
    <div class="hero-badges">
        <span class="badge">Devanagari</span>
        <span class="badge">Bangla</span>
        <span class="badge">Sanskrit</span>
        <span class="badge">Hindi</span>
        <span class="badge">CNN Classifier</span>
        <span class="badge">Tesseract OCR</span>
        <span class="badge">LLM Reconstruction</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#FF6B35;'
                'margin-bottom:1rem">⚙ System Configuration</p>',
                unsafe_allow_html=True)

    st.markdown("**Language Hint**")
    lang_sel = st.selectbox(
        "Script override",
        ["Auto Detect", "Devanagari", "Bangla"],
        label_visibility="collapsed",
    )
    language_hint: Optional[str] = {"Auto Detect": None,
                                    "Devanagari": "devanagari",
                                    "Bangla": "bangla"}[lang_sel]

    st.markdown("**Confidence Alert Threshold**")
    conf_threshold = st.slider("", 0.0, 1.0, 0.5, 0.05,
                               label_visibility="collapsed")

    st.markdown("**LLM API Keys**")
    st.caption("Required for Agent 4 (Linguistic Reconstruction)")
    anthropic_key = st.text_input("Anthropic API Key", type="password",
                                  placeholder="sk-ant-…")
    openai_key    = st.text_input("OpenAI API Key",    type="password",
                                  placeholder="sk-…")

    st.divider()

    st.markdown("""
    <div style="font-size:.78rem;color:#555;line-height:1.8">
    <b style="color:#888">Agent Pipeline</b><br>
    🔧 Image Restoration<br>
    🔍 Script Detection<br>
    📝 OCR Extraction<br>
    🧠 Linguistic Reconstruction
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Supported: JPG · PNG · TIFF · BMP (max 10 MB)")

# ── Upload tabs ───────────────────────────────────────────────────────────────
single_tab, batch_tab = st.tabs(["📄  Single Manuscript", "📚  Batch Processing"])

with single_tab:
    st.markdown('<div class="section-header">UPLOAD MANUSCRIPT IMAGE</div>',
                unsafe_allow_html=True)

    single_file = st.file_uploader(
        "Drag and drop a manuscript image or click to browse",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if single_file:
        pil_img = Image.open(single_file).convert("RGB")
        pr_col, meta_col = st.columns([2, 1], gap="large")

        with pr_col:
            st.markdown('<div class="img-card"><div class="img-card-label">'
                        'Uploaded Manuscript</div>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with meta_col:
            chips = (
                _metric_chip("Width",  f"{pil_img.width}px")
                + _metric_chip("Height", f"{pil_img.height}px")
                + _metric_chip("File size", f"{single_file.size // 1024} KB")
                + _metric_chip("Format", single_file.type.split("/")[-1].upper())
            )
            st.markdown(f'<div class="metric-row">{chips}</div>',
                        unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:1rem;font-size:.8rem;color:#555;line-height:1.9">
            <b style="color:#888">Processing Steps</b><br>
            🔧 Denoise + deskew + threshold<br>
            🔍 CNN script classification<br>
            📝 Tesseract OCR extraction<br>
            🧠 LLM linguistic reconstruction
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button("▶  Run Intelligence Pipeline",
                                type="primary", use_container_width=True)

        if run_btn:
            st.markdown('<div class="section-header">AI PROCESSING PIPELINE</div>',
                        unsafe_allow_html=True)
            pipeline_slot = st.empty()
            progress_slot = st.empty()

            # draw idle pipeline
            pipeline_slot.markdown(
                _render_pipeline(["pending"]*4, [""]*4),
                unsafe_allow_html=True,
            )

            with st.spinner(""):
                try:
                    result = _execute_pipeline(
                        image_bytes=single_file.getvalue(),
                        filename=single_file.name,
                        language_hint=language_hint,
                        anthropic_key=anthropic_key or None,
                        openai_key=openai_key or None,
                        pipeline_slot=pipeline_slot,
                        progress_slot=progress_slot,
                    )
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    st.stop()

            st.success(
                f"✅ All 4 agents completed in **{result['duration_ms']:.0f} ms** — "
                f"Script: **{result['script'].capitalize()}**"
            )
            st.divider()
            _render_results(result, single_file.getvalue())

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;
                    border:2px dashed #1e2540;border-radius:12px;
                    background:#0D1120;color:#444;margin-top:1rem">
            <div style="font-size:3rem;margin-bottom:1rem">📜</div>
            <div style="font-size:1rem;color:#666;margin-bottom:.5rem">
                Upload a manuscript image to begin analysis
            </div>
            <div style="font-size:.82rem;color:#444">
                Supported: Devanagari · Bangla manuscripts<br>
                JPG · PNG · TIFF · BMP formats accepted
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Batch tab ─────────────────────────────────────────────────────────────────
with batch_tab:
    st.markdown('<div class="section-header">BATCH MANUSCRIPT PROCESSING</div>',
                unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple manuscript images",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_up",
    )

    if batch_files:
        st.info(f"**{len(batch_files)}** manuscript(s) queued for batch processing.")
        bc, _ = st.columns([1, 3])
        with bc:
            batch_btn = st.button(f"▶  Process {len(batch_files)} Manuscripts",
                                  type="secondary", use_container_width=True)

        if batch_btn:
            batch_results = []
            prog = st.progress(0, text="Starting batch…")
            for i, f in enumerate(batch_files):
                prog.progress((i) / len(batch_files), text=f"Processing {f.name}…")
                pl = st.empty()
                ps = st.empty()
                try:
                    res = _execute_pipeline(
                        image_bytes=f.getvalue(),
                        filename=f.name,
                        language_hint=language_hint,
                        anthropic_key=anthropic_key or None,
                        openai_key=openai_key or None,
                        pipeline_slot=pl,
                        progress_slot=ps,
                    )
                    batch_results.append(
                        {k: v for k, v in res.items() if not k.startswith("_")}
                    )
                except Exception as exc:
                    batch_results.append({"file": f.name, "error": str(exc)})
                pl.empty(); ps.empty()

            prog.progress(100, text=f"Batch complete — {len(batch_files)} processed")
            st.success(f"Batch complete: {len(batch_results)} manuscripts processed.")
            st.download_button(
                "📥 Download Batch Results (JSON)",
                data=json.dumps(batch_results, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="batch_manuscripts.json",
                mime="application/json",
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
            border-top:1px solid #1e2540">
    <span style="font-size:.78rem;color:#333;letter-spacing:.3px">
        Multi-Agent Manuscript Intelligence System &nbsp;·&nbsp;
        Degraded Devanagari &amp; Bangla Script Identification &nbsp;·&nbsp;
        4-Agent AI Pipeline
    </span>
</div>
""", unsafe_allow_html=True)
