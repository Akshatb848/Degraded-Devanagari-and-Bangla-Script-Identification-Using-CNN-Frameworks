"""
Agent 5: LLM Correction Agent
Uses Claude/GPT to correct OCR errors using language context and reasoning.

Corrections applied:
- Spelling errors introduced by OCR
- Missing matras (vowel marks) in Devanagari
- Incorrect character substitutions common in degraded documents
- Sentence-level grammar reconstruction
"""
import time
from typing import List, Dict, Tuple, Optional
import structlog

from agents.state import PipelineState

logger = structlog.get_logger()

# Script-specific correction prompts
CORRECTION_PROMPTS = {
    "devanagari": """You are an expert in Devanagari script and Hindi/Sanskrit text correction.
The following text was extracted from a degraded historical document using OCR and may contain errors.

Common OCR errors in Devanagari:
- Missing or incorrect matras (vowel diacritics)
- Confusion between similar characters (ण/न, ष/श, व/ब)
- Merged or split words
- Missing chandrabindu or anusvara

Raw OCR text:
{raw_text}

Please:
1. Correct all OCR errors while preserving the original meaning
2. Restore missing diacritical marks
3. Fix word boundaries
4. List each correction you made

Respond in JSON format:
{{
  "corrected_text": "<corrected text>",
  "corrections": ["<correction 1>", "<correction 2>", ...],
  "reasoning": "<brief explanation of major corrections>"
}}""",

    "bangla": """You are an expert in Bangla (Bengali) script and text correction.
The following text was extracted from a degraded document using OCR and may contain errors.

Common OCR errors in Bangla:
- Confusion between similar glyphs (ব/ভ, ড/ড়, র/ৰ)
- Missing hasanta or anusvar
- Incorrect juktakshara (conjunct consonants)
- Merged or split words

Raw OCR text:
{raw_text}

Please:
1. Correct all OCR errors while preserving the original meaning
2. Restore missing diacritical marks and conjuncts
3. Fix word boundaries
4. List each correction you made

Respond in JSON format:
{{
  "corrected_text": "<corrected text>",
  "corrections": ["<correction 1>", "<correction 2>", ...],
  "reasoning": "<brief explanation of major corrections>"
}}""",

    "unknown": """You are an expert in Indic scripts (Devanagari and Bangla) text correction.
The following text was extracted from a degraded document using OCR and may contain errors.

Raw OCR text:
{raw_text}

Please correct OCR errors, restore missing characters, and fix word boundaries.

Respond in JSON format:
{{
  "corrected_text": "<corrected text>",
  "corrections": ["<correction 1>", "<correction 2>", ...],
  "reasoning": "<brief explanation>"
}}""",
}


def llm_correction_agent(state: PipelineState) -> PipelineState:
    """
    Use an LLM to correct OCR output.

    Input:  state.raw_text, state.script
    Output: state.corrected_text, state.corrections_made, state.reasoning
    """
    start = time.time()
    request_id = state["request_id"]
    logger.info("agent_start", agent="LLMCorrectionAgent", request_id=request_id)

    raw_text = state.get("raw_text", "")
    if not raw_text or not raw_text.strip():
        state["corrected_text"] = ""
        state["corrections_made"] = []
        state["reasoning"] = "No text to correct"
        _record_agent_status(state, "LLMCorrectionAgent", "skipped", output={"reason": "empty_text"})
        return state

    try:
        script = state.get("script", "unknown")
        corrected_text, corrections, reasoning = _run_llm_correction(raw_text, script)

        elapsed = (time.time() - start) * 1000

        state["corrected_text"] = corrected_text
        state["corrections_made"] = corrections
        state["reasoning"] = reasoning

        _record_agent_status(
            state,
            "LLMCorrectionAgent",
            "completed",
            output={
                "corrections_count": len(corrections),
                "text_changed": corrected_text != raw_text,
            },
            processing_time_ms=elapsed,
        )

        logger.info(
            "agent_complete",
            agent="LLMCorrectionAgent",
            request_id=request_id,
            corrections_count=len(corrections),
            elapsed_ms=elapsed,
        )

    except Exception as e:
        logger.error("agent_error", agent="LLMCorrectionAgent", error=str(e))
        state["corrected_text"] = raw_text  # Fall back to raw text
        state["corrections_made"] = []
        state["reasoning"] = f"LLM correction failed: {str(e)}"
        _record_agent_status(state, "LLMCorrectionAgent", "failed", error=str(e))
        _add_error(state, f"LLMCorrectionAgent: {e}")

    return state


def _run_llm_correction(
    raw_text: str,
    script: str,
) -> Tuple[str, List[str], str]:
    """
    Call LLM for OCR correction.
    Tries Anthropic Claude first, falls back to OpenAI.
    """
    try:
        return _anthropic_correction(raw_text, script)
    except Exception as e:
        logger.warning("anthropic_correction_failed", error=str(e))

    try:
        return _openai_correction(raw_text, script)
    except Exception as e:
        logger.warning("openai_correction_failed", error=str(e))

    # Last resort: return raw text
    return raw_text, [], "LLM unavailable - returning raw OCR text"


def _anthropic_correction(raw_text: str, script: str) -> Tuple[str, List[str], str]:
    """Use Anthropic Claude for correction."""
    import anthropic
    import json
    from app.core.config import settings

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = CORRECTION_PROMPTS.get(script, CORRECTION_PROMPTS["unknown"])
    message_content = prompt.format(raw_text=raw_text)

    response = client.messages.create(
        model=settings.default_llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        messages=[{"role": "user", "content": message_content}],
    )

    response_text = response.content[0].text
    return _parse_llm_response(response_text, raw_text)


def _openai_correction(raw_text: str, script: str) -> Tuple[str, List[str], str]:
    """Use OpenAI GPT for correction."""
    from openai import OpenAI
    import json
    from app.core.config import settings

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = CORRECTION_PROMPTS.get(script, CORRECTION_PROMPTS["unknown"])
    message_content = prompt.format(raw_text=raw_text)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        messages=[{"role": "user", "content": message_content}],
        response_format={"type": "json_object"},
    )

    response_text = response.choices[0].message.content
    return _parse_llm_response(response_text, raw_text)


def _parse_llm_response(
    response_text: str,
    raw_text: str,
) -> Tuple[str, List[str], str]:
    """Parse JSON response from LLM."""
    import json

    try:
        # Try direct JSON parse
        data = json.loads(response_text)
        corrected = data.get("corrected_text", raw_text)
        corrections = data.get("corrections", [])
        reasoning = data.get("reasoning", "")
        return corrected, corrections, reasoning
    except json.JSONDecodeError:
        # Try to extract JSON block from markdown
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return (
                    data.get("corrected_text", raw_text),
                    data.get("corrections", []),
                    data.get("reasoning", ""),
                )
            except Exception:
                pass

    return raw_text, [], f"Could not parse LLM response"


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
