"""
Agent 1: Script Detection Agent
Uses trained CNN models to classify Devanagari or Bangla script.
"""
import time
from typing import Dict, Any
import structlog

from agents.state import PipelineState

logger = structlog.get_logger()


def script_detection_agent(state: PipelineState) -> PipelineState:
    """
    Classify the script in the document image.

    Input:  state.image_bytes
    Output: state.script, state.script_confidence, state.script_model_used
    """
    start = time.time()
    request_id = state["request_id"]

    logger.info("agent_start", agent="ScriptDetectionAgent", request_id=request_id)

    try:
        from models.cnn_classifier import ScriptClassifier

        classifier = ScriptClassifier.get_instance()

        # Use language hint to override if provided
        if state.get("language_hint") and state["language_hint"] in ("devanagari", "bangla"):
            script = state["language_hint"]
            confidence = 1.0
            model_used = "hint_override"
            logger.info(
                "script_hint_used",
                script=script,
                request_id=request_id,
            )
        else:
            script, confidence, model_used = classifier.predict(
                state["image_bytes"],
                model_name="ensemble",
            )

        elapsed = (time.time() - start) * 1000

        state["script"] = script
        state["script_confidence"] = confidence
        state["script_model_used"] = model_used

        _record_agent_status(
            state,
            agent_name="ScriptDetectionAgent",
            status="completed",
            output={"script": script, "confidence": confidence, "model": model_used},
            processing_time_ms=elapsed,
        )

        logger.info(
            "agent_complete",
            agent="ScriptDetectionAgent",
            request_id=request_id,
            script=script,
            confidence=confidence,
            elapsed_ms=elapsed,
        )

    except Exception as e:
        logger.error("agent_error", agent="ScriptDetectionAgent", request_id=request_id, error=str(e))
        state["script"] = "unknown"
        state["script_confidence"] = 0.0
        state["script_model_used"] = "failed"
        _record_agent_status(state, "ScriptDetectionAgent", "failed", error=str(e))
        _add_error(state, f"ScriptDetectionAgent: {e}")

    return state


def _record_agent_status(
    state: PipelineState,
    agent_name: str,
    status: str,
    output: Dict[str, Any] = None,
    error: str = None,
    processing_time_ms: float = None,
) -> None:
    if state.get("agent_statuses") is None:
        state["agent_statuses"] = []
    state["agent_statuses"].append({
        "agent_name": agent_name,
        "status": status,
        "output": output,
        "error": error,
        "processing_time_ms": processing_time_ms,
    })


def _add_error(state: PipelineState, error: str) -> None:
    if state.get("errors") is None:
        state["errors"] = []
    state["errors"].append(error)
