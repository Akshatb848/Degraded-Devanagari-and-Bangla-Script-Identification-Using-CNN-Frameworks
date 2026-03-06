"""
LangGraph Multi-Agent Orchestrator for AIOCR.

Graph topology:
  START
    └─► script_detection
          └─► image_restoration
                └─► text_detection
                      └─► char_recognition
                            └─► llm_correction
                                  └─► knowledge_retrieval
                                        └─► output_formatting
                                              └─► END

Parallel branch (future):
  script_detection + image_restoration run in parallel,
  then merge before text_detection.
"""
import asyncio
from typing import Any, Dict, Optional
import structlog
from langgraph.graph import StateGraph, START, END

from agents.state import PipelineState
from agents.script_detection_agent import script_detection_agent
from agents.image_restoration_agent import image_restoration_agent
from agents.text_detection_agent import text_detection_agent
from agents.char_recognition_agent import char_recognition_agent
from agents.llm_correction_agent import llm_correction_agent
from agents.knowledge_retrieval_agent import knowledge_retrieval_agent
from agents.output_formatting_agent import output_formatting_agent

logger = structlog.get_logger()


def _build_graph() -> StateGraph:
    """Construct the LangGraph state machine for the OCR pipeline."""

    graph = StateGraph(PipelineState)

    # Register all 7 agent nodes
    graph.add_node("script_detection", script_detection_agent)
    graph.add_node("image_restoration", image_restoration_agent)
    graph.add_node("text_detection", text_detection_agent)
    graph.add_node("char_recognition", char_recognition_agent)
    graph.add_node("llm_correction", llm_correction_agent)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_agent)
    graph.add_node("output_formatting", output_formatting_agent)

    # Sequential pipeline edges
    graph.add_edge(START, "script_detection")
    graph.add_edge("script_detection", "image_restoration")
    graph.add_edge("image_restoration", "text_detection")
    graph.add_edge("text_detection", "char_recognition")
    graph.add_edge("char_recognition", "llm_correction")
    graph.add_edge("llm_correction", "knowledge_retrieval")
    graph.add_edge("knowledge_retrieval", "output_formatting")
    graph.add_edge("output_formatting", END)

    return graph.compile()


def _build_ocr_graph() -> StateGraph:
    """
    Lightweight OCR-only graph (no image restoration).
    Used by the /ocr endpoint.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("script_detection", script_detection_agent)
    graph.add_node("text_detection", text_detection_agent)
    graph.add_node("char_recognition", char_recognition_agent)
    graph.add_node("llm_correction", llm_correction_agent)
    graph.add_node("output_formatting", output_formatting_agent)

    graph.add_edge(START, "script_detection")
    graph.add_edge("script_detection", "text_detection")
    graph.add_edge("text_detection", "char_recognition")
    graph.add_edge("char_recognition", "llm_correction")
    graph.add_edge("llm_correction", "output_formatting")
    graph.add_edge("output_formatting", END)

    return graph.compile()


# Compile graphs once at module load
_full_pipeline_graph = _build_graph()
_ocr_pipeline_graph = _build_ocr_graph()


class AIRCOrchestrator:
    """
    High-level orchestrator that executes the multi-agent LangGraph pipeline.
    """

    async def run(
        self,
        image_bytes: bytes,
        request_id: str,
        language_hint: Optional[str] = None,
        enable_restoration: bool = True,
        enable_rag: bool = True,
        include_annotated_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full 7-agent pipeline.
        Returns the final_output dict from OutputFormattingAgent.
        """
        initial_state: PipelineState = {
            "request_id": request_id,
            "image_bytes": image_bytes,
            "language_hint": language_hint,
            "enable_restoration": enable_restoration,
            "enable_rag": enable_rag,
            "include_annotated_image": include_annotated_image,
            # All agent outputs initialized to None
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

        logger.info("orchestrator_run_start", request_id=request_id)

        # Run in thread pool to avoid blocking the event loop
        # (LangGraph sync nodes)
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: _full_pipeline_graph.invoke(initial_state),
        )

        output = final_state.get("final_output") or {}
        output["request_id"] = request_id

        # Inject overall_confidence if not set by output agent
        if "overall_confidence" not in output:
            output["overall_confidence"] = final_state.get("overall_confidence", 0.0)

        logger.info(
            "orchestrator_run_complete",
            request_id=request_id,
            script=output.get("script"),
            confidence=output.get("overall_confidence"),
        )

        return output


async def run_ocr_pipeline(
    image_bytes: bytes,
    script_hint: Optional[str] = None,
    apply_correction: bool = True,
    include_bboxes: bool = True,
) -> Dict[str, Any]:
    """
    Lightweight OCR-only pipeline for the /ocr endpoint.
    Skips image restoration and RAG agents.
    """
    import uuid

    request_id = str(uuid.uuid4())
    initial_state: PipelineState = {
        "request_id": request_id,
        "image_bytes": image_bytes,
        "language_hint": script_hint,
        "enable_restoration": False,
        "enable_rag": False,
        "include_annotated_image": include_bboxes,
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

    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(
        None,
        lambda: _ocr_pipeline_graph.invoke(initial_state),
    )

    output = final_state.get("final_output") or {}
    regions = output.get("text_regions", [])
    bboxes = [r["bounding_box"] for r in regions]

    return {
        "script": output.get("script", "unknown"),
        "raw_text": final_state.get("raw_text", ""),
        "corrected_text": output.get("corrected_text", ""),
        "confidence": output.get("overall_confidence", 0.0),
        "text_regions": regions,
        "bounding_boxes": bboxes,
        "language": output.get("language", "Unknown"),
    }
