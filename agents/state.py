"""
LangGraph pipeline state definition.
Shared state object passed between all 7 agents.
"""
from typing import TypedDict, List, Optional, Dict, Any


class PipelineState(TypedDict):
    # Input
    request_id: str
    image_bytes: bytes
    language_hint: Optional[str]
    enable_restoration: bool
    enable_rag: bool
    include_annotated_image: bool

    # Agent 1 - Script Detection
    script: Optional[str]
    script_confidence: Optional[float]
    script_model_used: Optional[str]

    # Agent 2 - Image Restoration
    restored_image_bytes: Optional[bytes]
    restoration_applied: Optional[List[str]]
    quality_before: Optional[float]
    quality_after: Optional[float]

    # Agent 3 - Text Detection
    text_regions_raw: Optional[List[Dict[str, Any]]]  # [{bbox, region_image_bytes}, ...]

    # Agent 4 - Character Recognition
    raw_text: Optional[str]
    raw_text_per_region: Optional[List[str]]
    ocr_confidence: Optional[float]

    # Agent 5 - LLM Correction
    corrected_text: Optional[str]
    corrections_made: Optional[List[str]]
    reasoning: Optional[str]

    # Agent 6 - Knowledge Retrieval
    validated_text: Optional[str]
    retrieved_context: Optional[str]
    rag_corrections: Optional[List[str]]

    # Agent 7 - Output Formatting
    final_output: Optional[Dict[str, Any]]
    annotated_image_base64: Optional[str]

    # Meta
    agent_statuses: Optional[List[Dict[str, Any]]]
    errors: Optional[List[str]]
    overall_confidence: Optional[float]
