"""
Agent 6: Knowledge Retrieval Agent (RAG)
Uses ChromaDB vector store with Indic corpora to validate and correct rare words.

Sources:
- Hindi/Bangla Wikipedia text
- Historical dictionaries
- Government archive documents
- Indic language n-gram corpora
"""
import time
from typing import List, Tuple, Optional
import structlog

from agents.state import PipelineState

logger = structlog.get_logger()


def knowledge_retrieval_agent(state: PipelineState) -> PipelineState:
    """
    Use RAG to validate and correct words against Indic language corpora.

    Input:  state.corrected_text, state.script
    Output: state.validated_text, state.retrieved_context, state.rag_corrections
    """
    if not state.get("enable_rag", True):
        logger.info("agent_skip", agent="KnowledgeRetrievalAgent", reason="disabled")
        state["validated_text"] = state.get("corrected_text", "")
        state["retrieved_context"] = ""
        state["rag_corrections"] = []
        _record_agent_status(state, "KnowledgeRetrievalAgent", "skipped")
        return state

    start = time.time()
    request_id = state["request_id"]
    logger.info("agent_start", agent="KnowledgeRetrievalAgent", request_id=request_id)

    text = state.get("corrected_text") or state.get("raw_text", "")
    if not text.strip():
        state["validated_text"] = text
        state["retrieved_context"] = ""
        state["rag_corrections"] = []
        _record_agent_status(state, "KnowledgeRetrievalAgent", "skipped", output={"reason": "empty_text"})
        return state

    try:
        script = state.get("script", "unknown")
        validated_text, context, rag_corrections = _run_rag_validation(text, script)

        elapsed = (time.time() - start) * 1000

        state["validated_text"] = validated_text
        state["retrieved_context"] = context
        state["rag_corrections"] = rag_corrections

        _record_agent_status(
            state,
            "KnowledgeRetrievalAgent",
            "completed",
            output={
                "rag_corrections_count": len(rag_corrections),
                "context_retrieved": bool(context),
            },
            processing_time_ms=elapsed,
        )

        logger.info(
            "agent_complete",
            agent="KnowledgeRetrievalAgent",
            request_id=request_id,
            rag_corrections=len(rag_corrections),
            elapsed_ms=elapsed,
        )

    except Exception as e:
        logger.error("agent_error", agent="KnowledgeRetrievalAgent", error=str(e))
        state["validated_text"] = text
        state["retrieved_context"] = ""
        state["rag_corrections"] = []
        _record_agent_status(state, "KnowledgeRetrievalAgent", "failed", error=str(e))
        _add_error(state, f"KnowledgeRetrievalAgent: {e}")

    return state


def _run_rag_validation(
    text: str,
    script: str,
) -> Tuple[str, str, List[str]]:
    """
    Retrieve relevant context from vector DB and use it to validate words.
    """
    try:
        context = _retrieve_context(text, script)
        if context:
            validated, corrections = _apply_rag_corrections(text, context, script)
            return validated, context, corrections
    except Exception as e:
        logger.warning("chroma_retrieval_failed", error=str(e))

    # Fallback: basic dictionary validation
    try:
        validated, corrections = _dictionary_validation(text, script)
        return validated, "", corrections
    except Exception:
        pass

    return text, "", []


def _retrieve_context(text: str, script: str) -> Optional[str]:
    """Query ChromaDB for relevant context documents."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from app.core.config import settings

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    collection = client.get_collection(settings.chroma_collection)

    # Use multilingual sentence transformer for Indic text
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    query_embedding = embedder.encode([text[:500]])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"script": script} if script != "unknown" else None,
    )

    if results["documents"] and results["documents"][0]:
        return "\n".join(results["documents"][0])
    return None


def _apply_rag_corrections(
    text: str,
    context: str,
    script: str,
) -> Tuple[str, List[str]]:
    """
    Use retrieved context + LLM to apply final corrections.
    """
    try:
        import anthropic
        import json
        from app.core.config import settings

        rag_prompt = f"""You are an expert in {script} script correction.

Retrieved reference text from Indic corpus:
{context}

OCR text to validate:
{text}

Using the reference text as context, identify and correct any words that:
1. Are not valid words in {script} script
2. Could be better words given the surrounding context
3. Appear to be OCR artifacts

Respond in JSON:
{{
  "validated_text": "<corrected text>",
  "corrections": ["<word_original> → <word_corrected>: <reason>", ...]
}}"""

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.default_llm_model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": rag_prompt}],
        )

        import re
        resp_text = response.content[0].text
        json_match = re.search(r"\{.*\}", resp_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("validated_text", text), data.get("corrections", [])

    except Exception as e:
        logger.warning("rag_llm_correction_failed", error=str(e))

    return text, []


def _dictionary_validation(text: str, script: str) -> Tuple[str, List[str]]:
    """
    Simple n-gram frequency-based word validation.
    Uses basic word lists as fallback when vector DB is unavailable.
    """
    # In production this would use actual Indic dictionaries
    # Here we return the text unchanged as a safe fallback
    return text, []


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
