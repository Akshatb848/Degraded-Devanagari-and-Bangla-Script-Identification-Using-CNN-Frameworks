"""
LLM Corrector
─────────────
Post-processes raw OCR text with a large language model specialized in
Indic manuscript restoration.

Supported providers (tried in order of availability):
  1. Anthropic Claude  — requires ANTHROPIC_API_KEY
  2. OpenAI GPT-4o     — requires OPENAI_API_KEY
  3. None              — returns raw text unchanged with a warning

Prompt templates use __TEXT__, __SCRIPT__, __LANG__ sentinels (never
curly-brace placeholders) so str.replace() never collides with any
content in the OCR text.

Returns a CorrectionResult dataclass.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── script → language hint mapping ───────────────────────────────────────────

_LANG_HINTS: dict[str, str] = {
    "devanagari": "Sanskrit / Hindi / Marathi",
    "bangla":     "Bangla (Bengali)",
    "unknown":    "Unknown Indic script",
}

# ── confidence label → numeric ────────────────────────────────────────────────

_CONF_MAP: dict[str, float] = {
    "high":   0.90,
    "medium": 0.60,
    "low":    0.30,
}

# ── master prompt template ────────────────────────────────────────────────────
# Sentinels:  __SCRIPT__  __LANG__  __TEXT__
# All curly braces inside belong to the output-format example; they are NOT
# Python format specifiers — safe because we use str.replace(), not str.format().

_MASTER_PROMPT = """\
You are an advanced AI system specialized in:
• Indic manuscript restoration • Devanagari and Bangla script recognition \
• OCR error correction • Sanskrit, Hindi, and Bangla linguistic reasoning

The input comes from an OCR system applied to degraded manuscript images.
The manuscripts may contain:
* faded ink          * broken characters      * merged characters
* missing characters * incorrect segmentation * noise from paper degradation
* OCR misinterpretations

Your goal is to reconstruct the most accurate readable text while preserving
the original script and language.

INPUT CONTEXT
Script detected by CNN classifier: __SCRIPT__
Possible language: __LANG__
Raw OCR output: __TEXT__

TASK
Analyze the OCR output and correct recognition errors.
Apply the following reasoning steps:
1. Detect OCR mistakes caused by:
   * broken glyphs            * merged characters
   * incorrect segmentation   * visually similar letters
2. Reconstruct incomplete words using linguistic context.
3. Use grammar patterns from Sanskrit, Hindi, and Bangla.
4. Remove OCR artifacts: random punctuation, non-script symbols, repeated fragments.
5. Preserve the original script exactly:
   * Do NOT convert Devanagari to Latin
   * Do NOT translate the text
   * Do NOT change the language
6. Maintain natural spacing and sentence flow.
7. If a word is uncertain, infer the most probable form using linguistic reasoning.
8. Avoid hallucination — only reconstruct what is likely present in the manuscript.

OUTPUT FORMAT
Return output in this structured format:

Corrected Text:
<corrected manuscript text here>

Confidence Level: High / Medium / Low

Explanation:
<brief explanation of corrections applied>

STRICT RULES
• Do NOT translate the text.  • Do NOT summarize the text.
• Do NOT add new content.     • Only correct OCR recognition errors.

Focus entirely on reconstructing the most accurate version of the manuscript text.\
"""

_DEFAULT_MODEL_CLAUDE = "claude-3-5-haiku-20241022"
_DEFAULT_MODEL_OPENAI = "gpt-4o-mini"


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class CorrectionResult:
    corrected_text:   str        = ""
    corrections:      List[str]  = field(default_factory=list)
    reasoning:        str        = ""
    confidence_label: str        = ""      # "High" | "Medium" | "Low"
    confidence_score: float      = 0.0     # numeric mapping of confidence_label
    model_used:       str        = "none"
    provider:         str        = "none"  # "anthropic" | "openai" | "none"
    skipped:          bool       = False
    warning:          str        = ""


# ── public API ────────────────────────────────────────────────────────────────

def correct_text(
    raw_text:    str,
    script:      str  = "unknown",
    anthropic_key: Optional[str] = None,
    openai_key:    Optional[str] = None,
) -> CorrectionResult:
    """
    Attempt LLM correction of *raw_text*.

    Keys are read from arguments first, then from environment variables
    ANTHROPIC_API_KEY / OPENAI_API_KEY.
    """
    if not raw_text.strip():
        return CorrectionResult(
            corrected_text=raw_text,
            skipped=True,
            warning="Empty input — skipped LLM correction.",
        )

    prompt = _build_prompt(raw_text, script)

    # ── try Anthropic ─────────────────────────────────────────────────────────
    akey = anthropic_key or os.getenv("ANTHROPIC_API_KEY", "")
    if akey:
        try:
            return _call_anthropic(prompt, akey)
        except Exception as exc:
            pass   # fall through to OpenAI

    # ── try OpenAI ────────────────────────────────────────────────────────────
    okey = openai_key or os.getenv("OPENAI_API_KEY", "")
    if okey:
        try:
            return _call_openai(prompt, okey)
        except Exception as exc:
            pass

    # ── no provider available ─────────────────────────────────────────────────
    return CorrectionResult(
        corrected_text=raw_text,
        skipped=True,
        warning="No LLM API key configured — returning raw OCR text.",
    )


# ── provider implementations ──────────────────────────────────────────────────

def _build_prompt(raw_text: str, script: str) -> str:
    lang = _LANG_HINTS.get(script, _LANG_HINTS["unknown"])
    return (
        _MASTER_PROMPT
        .replace("__SCRIPT__", script)
        .replace("__LANG__",   lang)
        .replace("__TEXT__",   raw_text)
    )


def _call_anthropic(prompt: str, api_key: str) -> CorrectionResult:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_DEFAULT_MODEL_CLAUDE,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_response = message.content[0].text
    parsed = _parse_response(raw_response)
    parsed.model_used = _DEFAULT_MODEL_CLAUDE
    parsed.provider   = "anthropic"
    return parsed


def _call_openai(prompt: str, api_key: str) -> CorrectionResult:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=_DEFAULT_MODEL_OPENAI,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.2,
    )
    raw_response = response.choices[0].message.content or ""
    parsed = _parse_response(raw_response)
    parsed.model_used = _DEFAULT_MODEL_OPENAI
    parsed.provider   = "openai"
    return parsed


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str) -> CorrectionResult:
    """
    Parse the structured-text output produced by the manuscript restoration prompt:

        Corrected Text:
        <text>

        Confidence Level: High / Medium / Low

        Explanation:
        <explanation>

    Falls back gracefully if the model returns something unexpected.
    """
    # ── extract Corrected Text ────────────────────────────────────────────────
    corrected_text = ""
    ct_match = re.search(
        r"Corrected\s+Text\s*:\s*\n(.*?)(?=\nConfidence\s+Level\s*:|$)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if ct_match:
        corrected_text = ct_match.group(1).strip()

    # ── extract Confidence Level ──────────────────────────────────────────────
    conf_label = ""
    conf_score = 0.0
    cl_match = re.search(
        r"Confidence\s+Level\s*:\s*(High|Medium|Low)",
        raw, re.IGNORECASE,
    )
    if cl_match:
        conf_label = cl_match.group(1).capitalize()
        conf_score = _CONF_MAP.get(conf_label.lower(), 0.0)

    # ── extract Explanation ───────────────────────────────────────────────────
    explanation = ""
    ex_match = re.search(
        r"Explanation\s*:\s*\n?(.*?)$",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if ex_match:
        explanation = ex_match.group(1).strip()

    # ── fallback: use raw response if nothing parsed ──────────────────────────
    warning = ""
    if not corrected_text:
        corrected_text = raw.strip()
        warning = "Could not parse structured response — using raw LLM reply."

    return CorrectionResult(
        corrected_text=corrected_text,
        reasoning=explanation,
        confidence_label=conf_label,
        confidence_score=conf_score,
        warning=warning,
    )
