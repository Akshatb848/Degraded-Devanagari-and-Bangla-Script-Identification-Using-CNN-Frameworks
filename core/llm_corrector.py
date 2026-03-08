"""
LLM Corrector
─────────────
Post-processes raw OCR text with a large language model.

Supported providers (tried in order of availability):
  1. Anthropic Claude  — requires ANTHROPIC_API_KEY
  2. OpenAI GPT-4o     — requires OPENAI_API_KEY
  3. None              — returns raw text unchanged with a warning

Prompt templates use the __TEXT__ sentinel (not {text}) so that
str.format() / str.replace() never trips on JSON examples embedded
in the prompt body.

Returns a CorrectionResult dataclass.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── prompt templates ──────────────────────────────────────────────────────────

_PROMPTS: dict[str, str] = {
    "devanagari": (
        "You are an expert in Devanagari script (Hindi/Sanskrit). "
        "The following text was extracted by OCR from a degraded manuscript. "
        "Correct OCR errors, fix broken words, and restore proper Unicode. "
        "Return ONLY a JSON object with exactly these keys:\n"
        '{"corrected_text":"<corrected>","corrections":["<change 1>","<change 2>"],"reasoning":"<brief>"}\n\n'
        "OCR text:\n__TEXT__"
    ),
    "bangla": (
        "You are an expert in Bangla (Bengali) script. "
        "The following text was extracted by OCR from a degraded manuscript. "
        "Correct OCR errors, fix broken words, and restore proper Unicode. "
        "Return ONLY a JSON object with exactly these keys:\n"
        '{"corrected_text":"<corrected>","corrections":["<change 1>","<change 2>"],"reasoning":"<brief>"}\n\n'
        "OCR text:\n__TEXT__"
    ),
    "unknown": (
        "You are an expert in Indic scripts. "
        "The following text was extracted by OCR from a degraded manuscript. "
        "Identify the script, correct OCR errors, and restore proper Unicode. "
        "Return ONLY a JSON object with exactly these keys:\n"
        '{"corrected_text":"<corrected>","corrections":["<change 1>","<change 2>"],"reasoning":"<brief>"}\n\n'
        "OCR text:\n__TEXT__"
    ),
}

_DEFAULT_MODEL_CLAUDE = "claude-3-5-haiku-20241022"
_DEFAULT_MODEL_OPENAI = "gpt-4o-mini"


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class CorrectionResult:
    corrected_text: str        = ""
    corrections:    List[str]  = field(default_factory=list)
    reasoning:      str        = ""
    model_used:     str        = "none"
    provider:       str        = "none"   # "anthropic" | "openai" | "none"
    skipped:        bool       = False
    warning:        str        = ""


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
    template = _PROMPTS.get(script, _PROMPTS["unknown"])
    return template.replace("__TEXT__", raw_text)


def _call_anthropic(prompt: str, api_key: str) -> CorrectionResult:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_DEFAULT_MODEL_CLAUDE,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_response = message.content[0].text
    parsed = _parse_json_response(raw_response)
    parsed.model_used = _DEFAULT_MODEL_CLAUDE
    parsed.provider   = "anthropic"
    return parsed


def _call_openai(prompt: str, api_key: str) -> CorrectionResult:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=_DEFAULT_MODEL_OPENAI,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    raw_response = response.choices[0].message.content or ""
    parsed = _parse_json_response(raw_response)
    parsed.model_used = _DEFAULT_MODEL_OPENAI
    parsed.provider   = "openai"
    return parsed


# ── JSON response parser ──────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> CorrectionResult:
    """
    Extract the JSON object from the LLM response.
    Handles markdown code fences and leading/trailing prose.
    """
    # Strip markdown code fence if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Extract first {...} block
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return CorrectionResult(
            corrected_text=raw,
            warning="Could not parse JSON from LLM response — returning raw reply.",
        )

    try:
        data = json.loads(match.group())
        return CorrectionResult(
            corrected_text=str(data.get("corrected_text", raw)),
            corrections=list(data.get("corrections", [])),
            reasoning=str(data.get("reasoning", "")),
        )
    except json.JSONDecodeError:
        return CorrectionResult(
            corrected_text=raw,
            warning="JSON decode error — returning raw LLM reply as corrected text.",
        )
