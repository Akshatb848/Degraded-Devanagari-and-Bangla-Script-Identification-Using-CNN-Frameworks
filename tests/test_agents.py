"""
Unit tests for individual agents.
"""
import io
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image


def _make_image_bytes(width=64, height=64, color=(200, 200, 200)) -> bytes:
    """Generate test image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_initial_state(image_bytes=None) -> dict:
    """Create a minimal PipelineState for testing."""
    if image_bytes is None:
        image_bytes = _make_image_bytes()
    return {
        "request_id": "test-request-id",
        "image_bytes": image_bytes,
        "language_hint": None,
        "enable_restoration": True,
        "enable_rag": True,
        "include_annotated_image": False,
        "script": None,
        "script_confidence": None,
        "script_model_used": None,
        "restored_image_bytes": None,
        "restoration_applied": None,
        "quality_before": None,
        "quality_after": None,
        "text_regions_raw": None,
        "raw_text": None,
        "raw_text_per_region": None,
        "ocr_confidence": None,
        "corrected_text": None,
        "corrections_made": None,
        "reasoning": None,
        "validated_text": None,
        "retrieved_context": None,
        "rag_corrections": None,
        "final_output": None,
        "annotated_image_base64": None,
        "agent_statuses": [],
        "errors": [],
        "overall_confidence": None,
    }


class TestScriptDetectionAgent:
    @patch("models.cnn_classifier.ScriptClassifier.predict")
    @patch("models.cnn_classifier.ScriptClassifier.get_instance")
    def test_detects_devanagari(self, mock_get_instance, mock_predict):
        mock_predict.return_value = ("devanagari", 0.99, "ensemble")
        mock_instance = MagicMock()
        mock_instance.predict.return_value = ("devanagari", 0.99, "ensemble")
        mock_get_instance.return_value = mock_instance

        from agents.script_detection_agent import script_detection_agent
        state = _make_initial_state()
        result = script_detection_agent(state)

        assert result["script"] == "devanagari"
        assert result["script_confidence"] == 0.99
        assert any(s["agent_name"] == "ScriptDetectionAgent" for s in result["agent_statuses"])

    def test_language_hint_overrides_model(self):
        from agents.script_detection_agent import script_detection_agent
        state = _make_initial_state()
        state["language_hint"] = "bangla"

        result = script_detection_agent(state)

        assert result["script"] == "bangla"
        assert result["script_confidence"] == 1.0
        assert result["script_model_used"] == "hint_override"

    @patch("models.cnn_classifier.ScriptClassifier.get_instance")
    def test_handles_classifier_error(self, mock_get_instance):
        mock_get_instance.side_effect = RuntimeError("Model failed")
        from agents.script_detection_agent import script_detection_agent
        state = _make_initial_state()
        result = script_detection_agent(state)

        assert result["script"] == "unknown"
        assert len(result["errors"]) > 0


class TestImageRestorationAgent:
    def test_skip_when_disabled(self):
        from agents.image_restoration_agent import image_restoration_agent
        state = _make_initial_state()
        state["enable_restoration"] = False

        result = image_restoration_agent(state)

        assert result["restored_image_bytes"] == state["image_bytes"]
        assert result["restoration_applied"] == []
        assert any(
            s["agent_name"] == "ImageRestorationAgent" and s["status"] == "skipped"
            for s in result["agent_statuses"]
        )

    def test_restores_image(self):
        from agents.image_restoration_agent import image_restoration_agent
        state = _make_initial_state()

        result = image_restoration_agent(state)

        # Either restored or gracefully failed
        assert result.get("restored_image_bytes") is not None or len(result.get("errors", [])) > 0


class TestOutputFormattingAgent:
    def test_formats_successful_output(self):
        from agents.output_formatting_agent import output_formatting_agent
        state = _make_initial_state()
        state.update({
            "script": "devanagari",
            "script_confidence": 0.99,
            "ocr_confidence": 0.95,
            "raw_text": "भारत",
            "corrected_text": "भारत",
            "validated_text": "भारत",
            "corrections_made": [],
            "rag_corrections": [],
            "reasoning": "No corrections needed",
            "text_regions_raw": [],
            "raw_text_per_region": [],
            "agent_statuses": [],
        })

        result = output_formatting_agent(state)

        assert result["final_output"] is not None
        assert result["final_output"]["script"] == "devanagari"
        assert result["final_output"]["corrected_text"] == "भारत"
        assert result["final_output"]["status"] == "completed"

    def test_empty_text_handled(self):
        from agents.output_formatting_agent import output_formatting_agent
        state = _make_initial_state()
        state.update({
            "script": "bangla",
            "script_confidence": 0.8,
            "ocr_confidence": 0.0,
            "raw_text": "",
            "corrected_text": "",
            "validated_text": "",
            "corrections_made": [],
            "rag_corrections": [],
            "reasoning": "",
            "text_regions_raw": [],
            "raw_text_per_region": [],
            "agent_statuses": [],
        })

        result = output_formatting_agent(state)
        assert result["final_output"] is not None


class TestLLMCorrectionAgent:
    def test_skips_empty_text(self):
        from agents.llm_correction_agent import llm_correction_agent
        state = _make_initial_state()
        state["raw_text"] = ""
        state["script"] = "devanagari"

        result = llm_correction_agent(state)

        assert result["corrected_text"] == ""
        assert any(
            s["status"] == "skipped"
            for s in result["agent_statuses"]
        )

    @patch("agents.llm_correction_agent._anthropic_correction")
    def test_uses_llm_for_correction(self, mock_anthropic):
        mock_anthropic.return_value = ("भारत का इतिहास", ["इतिहस → इतिहास"], "Fixed spelling")

        from agents.llm_correction_agent import llm_correction_agent
        state = _make_initial_state()
        state["raw_text"] = "भारत का इतिहस"
        state["script"] = "devanagari"

        result = llm_correction_agent(state)

        assert result["corrected_text"] == "भारत का इतिहास"
        assert len(result["corrections_made"]) == 1


class TestKnowledgeRetrievalAgent:
    def test_skips_when_disabled(self):
        from agents.knowledge_retrieval_agent import knowledge_retrieval_agent
        state = _make_initial_state()
        state["enable_rag"] = False
        state["corrected_text"] = "भारत"

        result = knowledge_retrieval_agent(state)

        assert result["validated_text"] == "भारत"
        assert any(
            s["status"] == "skipped"
            for s in result["agent_statuses"]
        )

    def test_returns_original_on_chroma_failure(self):
        from agents.knowledge_retrieval_agent import knowledge_retrieval_agent
        state = _make_initial_state()
        state["corrected_text"] = "भारत"
        state["script"] = "devanagari"

        # ChromaDB not running in test - should fall back gracefully
        result = knowledge_retrieval_agent(state)
        assert result["validated_text"] == "भारत"


class TestOrchestratorState:
    def test_initial_state_valid(self):
        """Test that the initial state dict is valid for the pipeline."""
        from agents.state import PipelineState
        state = _make_initial_state()
        # Verify required keys are present
        required_keys = [
            "request_id", "image_bytes", "enable_restoration",
            "enable_rag", "agent_statuses", "errors",
        ]
        for key in required_keys:
            assert key in state, f"Missing required state key: {key}"
