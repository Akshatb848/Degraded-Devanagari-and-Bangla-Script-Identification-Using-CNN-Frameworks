"""
Streamlit-compatible synchronous pipeline runner.
Runs all agents in sequence without FastAPI / Redis / PostgreSQL.
Each step yields a status update so Streamlit can display live progress.
"""
import io
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def _image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _compute_laplacian_variance(image_bytes: bytes) -> float:
    """Estimate image sharpness (proxy for quality)."""
    try:
        import cv2
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return min(float(cv2.Laplacian(img, cv2.CV_64F).var()) / 5000.0, 1.0)
    except Exception:
        return 0.0


# ── Agent 1 – Script Detection ────────────────────────────────────────────────

def run_script_detection(
    image_bytes: bytes,
    model_name: str = "ensemble",
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    if language_hint in ("devanagari", "bangla"):
        return {"script": language_hint, "confidence": 1.0, "model_used": "hint_override"}

    try:
        from models.cnn_classifier import ScriptClassifier
        clf = ScriptClassifier.get_instance()
        script, conf, model = clf.predict(image_bytes, model_name=model_name)
        return {"script": script, "confidence": conf, "model_used": model}
    except Exception as e:
        return {"script": "unknown", "confidence": 0.0, "model_used": "failed", "error": str(e)}


# ── Agent 2 – Image Restoration ───────────────────────────────────────────────

def run_image_restoration(image_bytes: bytes) -> Dict[str, Any]:
    enhancements: List[str] = []
    try:
        import cv2

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        quality_before = _compute_laplacian_variance(image_bytes)

        # Deskew
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) >= 100:
                angle = cv2.minAreaRect(coords)[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                if abs(angle) >= 0.5:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                    enhancements.append("deskew")
        except Exception:
            pass

        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        enhancements.append("denoising")

        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l_ch), a_ch, b_ch))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhancements.append("clahe_contrast")

        # Bicubic upscale ×2 (ESRGAN fallback)
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        enhancements.append("bicubic_2x")

        _, buf = cv2.imencode(".png", img)
        restored_bytes = buf.tobytes()
        quality_after = _compute_laplacian_variance(restored_bytes)

        return {
            "restored_bytes": restored_bytes,
            "enhancements": enhancements,
            "quality_before": round(quality_before, 4),
            "quality_after": round(quality_after, 4),
        }

    except Exception as e:
        return {
            "restored_bytes": image_bytes,
            "enhancements": [],
            "quality_before": 0.0,
            "quality_after": 0.0,
            "error": str(e),
        }


# ── Agent 3 – Text Detection ──────────────────────────────────────────────────

def run_text_detection(image_bytes: bytes) -> Dict[str, Any]:
    """Detect text regions via connected-components (no YOLOv8 model required)."""
    try:
        import cv2

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        h_img, w_img = img.shape[:2]
        for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 500:
                continue
            roi = img[y : y + bh, x : x + bw]
            _, buf = cv2.imencode(".png", roi)
            regions.append({
                "bbox": {"x": x, "y": y, "width": bw, "height": bh},
                "confidence": 0.75,
                "region_image_bytes": buf.tobytes(),
                "line_number": len(regions),
            })

        if not regions:
            # Fallback: whole image as one region
            _, buf = cv2.imencode(".png", img)
            regions = [{
                "bbox": {"x": 0, "y": 0, "width": w_img, "height": h_img},
                "confidence": 0.5,
                "region_image_bytes": buf.tobytes(),
                "line_number": 0,
            }]

        return {"regions": regions, "method": "connected_components"}

    except Exception as e:
        return {"regions": [], "method": "failed", "error": str(e)}


# ── Agent 4 – Character Recognition ──────────────────────────────────────────

def run_char_recognition(
    regions: List[Dict[str, Any]],
    script: str,
) -> Dict[str, Any]:
    """Try TrOCR; fall back to Tesseract; fall back to placeholder."""
    texts: List[str] = []
    method = "none"

    # Try Tesseract first (lighter, doesn't require GPU)
    try:
        import pytesseract
        lang_map = {"devanagari": "hin+san", "bangla": "ben", "unknown": "hin+ben"}
        lang = lang_map.get(script, "hin+ben")
        config = f"--oem 3 --psm 6 -l {lang}"
        for region in regions:
            pil_img = Image.open(io.BytesIO(region["region_image_bytes"])).convert("RGB")
            text = pytesseract.image_to_string(pil_img, config=config).strip()
            texts.append(text)
        method = "tesseract"
    except Exception:
        pass

    # Try TrOCR if Tesseract didn't work
    if not any(texts):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            model.eval()
            texts = []
            for region in regions:
                pil_img = Image.open(io.BytesIO(region["region_image_bytes"])).convert("RGB")
                px = processor(pil_img, return_tensors="pt").pixel_values
                with torch.no_grad():
                    ids = model.generate(px)
                texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
            method = "trocr"
        except Exception:
            texts = [""] * len(regions)

    raw_text = "\n".join(t for t in texts if t.strip())
    conf = 0.75 if raw_text.strip() else 0.0
    return {"raw_text": raw_text, "texts_per_region": texts, "confidence": conf, "method": method}


# ── Agent 5 – LLM Correction ─────────────────────────────────────────────────

PROMPTS = {
    "devanagari": (
        "You are an expert in Devanagari script. The following is raw OCR text from a degraded "
        "document — fix spelling errors, restore missing matras, and correct character "
        "substitutions (ण/न, ष/श). Respond ONLY in JSON:\n"
        '{"corrected_text":"...","corrections":["..."],"reasoning":"..."}\n\nOCR text:\n{text}'
    ),
    "bangla": (
        "You are an expert in Bangla script. The following is raw OCR text from a degraded "
        "document — fix glyph confusion (ব/ভ, ড/ড়), missing hasanta, incorrect conjuncts. "
        "Respond ONLY in JSON:\n"
        '{"corrected_text":"...","corrections":["..."],"reasoning":"..."}\n\nOCR text:\n{text}'
    ),
    "unknown": (
        "You are an expert in Indic scripts. Fix OCR errors in the text below. "
        "Respond ONLY in JSON:\n"
        '{"corrected_text":"...","corrections":["..."],"reasoning":"..."}\n\nOCR text:\n{text}'
    ),
}


def run_llm_correction(
    raw_text: str,
    script: str,
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
) -> Dict[str, Any]:
    if not raw_text.strip():
        return {"corrected_text": "", "corrections": [], "reasoning": "No text to correct"}

    prompt = PROMPTS.get(script, PROMPTS["unknown"]).format(text=raw_text)

    # Try Anthropic
    if anthropic_key:
        try:
            import anthropic, json, re
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_resp = resp.content[0].text
            m = re.search(r"\{.*\}", raw_resp, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return {
                    "corrected_text": data.get("corrected_text", raw_text),
                    "corrections": data.get("corrections", []),
                    "reasoning": data.get("reasoning", ""),
                    "model": "claude-sonnet-4-6",
                }
        except Exception:
            pass

    # Try OpenAI
    if openai_key:
        try:
            from openai import OpenAI
            import json, re
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2048,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "corrected_text": data.get("corrected_text", raw_text),
                "corrections": data.get("corrections", []),
                "reasoning": data.get("reasoning", ""),
                "model": "gpt-4o",
            }
        except Exception:
            pass

    return {
        "corrected_text": raw_text,
        "corrections": [],
        "reasoning": "LLM unavailable — displaying raw OCR text",
        "model": "none",
    }


# ── Agent 6 – Knowledge Retrieval (lightweight) ───────────────────────────────

def run_knowledge_retrieval(
    corrected_text: str,
    script: str,
) -> Dict[str, Any]:
    """
    Lightweight RAG: tries ChromaDB; falls back to no-op.
    In Streamlit Cloud, ChromaDB won't be running — this is a graceful no-op.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        client = chromadb.HttpClient(host="localhost", port=8001)
        collection = client.get_collection("indic_corpus")
        embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        emb = embedder.encode([corrected_text[:500]])[0].tolist()
        results = collection.query(query_embeddings=[emb], n_results=2)
        context = "\n".join(results["documents"][0]) if results["documents"][0] else ""
        return {"validated_text": corrected_text, "context": context, "rag_corrections": []}
    except Exception:
        return {"validated_text": corrected_text, "context": "", "rag_corrections": []}


# ── Agent 7 – Annotated Image ─────────────────────────────────────────────────

def create_annotated_image(
    image_bytes: bytes,
    regions: List[Dict[str, Any]],
) -> Optional[bytes]:
    try:
        import cv2
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        for i, region in enumerate(regions):
            bbox = region["bbox"]
            x, y = bbox["x"], bbox["y"]
            w, h = bbox["width"], bbox["height"]
            conf = region.get("confidence", 0.75)
            color = (0, 200, 0) if conf >= 0.8 else ((0, 200, 200) if conf >= 0.5 else (0, 0, 200))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"L{i+1}", (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()
    except Exception:
        return None


# ── Full synchronous pipeline ─────────────────────────────────────────────────

def run_full_pipeline(
    image_bytes: bytes,
    language_hint: Optional[str] = None,
    enable_restoration: bool = True,
    enable_rag: bool = False,
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Execute all 7 agents sequentially and return a combined result dict.
    `progress_callback(step, total, label)` is called between each agent.
    """

    def _progress(step: int, label: str):
        if progress_callback:
            progress_callback(step, 7, label)

    result: Dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "agent_timings": {},
    }

    # 1 — Script Detection
    _progress(1, "Agent 1/7 — Script Detection")
    t0 = time.time()
    det = run_script_detection(image_bytes, language_hint=language_hint)
    result["script"] = det["script"]
    result["script_confidence"] = det["confidence"]
    result["script_model"] = det.get("model_used", "")
    result["agent_timings"]["ScriptDetection"] = round((time.time() - t0) * 1000, 1)

    # 2 — Image Restoration
    _progress(2, "Agent 2/7 — Image Restoration")
    t0 = time.time()
    if enable_restoration:
        rest = run_image_restoration(image_bytes)
        working_bytes = rest["restored_bytes"]
        result["restoration"] = {
            "enhancements": rest["enhancements"],
            "quality_before": rest["quality_before"],
            "quality_after": rest["quality_after"],
        }
    else:
        working_bytes = image_bytes
        result["restoration"] = {"enhancements": [], "quality_before": 0.0, "quality_after": 0.0}
    result["restored_image_bytes"] = working_bytes
    result["agent_timings"]["ImageRestoration"] = round((time.time() - t0) * 1000, 1)

    # 3 — Text Detection
    _progress(3, "Agent 3/7 — Text Detection")
    t0 = time.time()
    td = run_text_detection(working_bytes)
    regions = td.get("regions", [])
    result["regions"] = regions
    result["text_detection_method"] = td.get("method", "")
    result["agent_timings"]["TextDetection"] = round((time.time() - t0) * 1000, 1)

    # 4 — Character Recognition
    _progress(4, "Agent 4/7 — Character Recognition (OCR)")
    t0 = time.time()
    ocr = run_char_recognition(regions, result["script"])
    result["raw_text"] = ocr["raw_text"]
    result["ocr_confidence"] = ocr["confidence"]
    result["ocr_method"] = ocr.get("method", "")
    result["texts_per_region"] = ocr.get("texts_per_region", [])
    result["agent_timings"]["CharRecognition"] = round((time.time() - t0) * 1000, 1)

    # 5 — LLM Correction
    _progress(5, "Agent 5/7 — LLM Correction")
    t0 = time.time()
    llm = run_llm_correction(
        result["raw_text"],
        result["script"],
        anthropic_key=anthropic_key,
        openai_key=openai_key,
    )
    result["corrected_text"] = llm["corrected_text"]
    result["corrections"] = llm["corrections"]
    result["reasoning"] = llm["reasoning"]
    result["llm_model"] = llm.get("model", "none")
    result["agent_timings"]["LLMCorrection"] = round((time.time() - t0) * 1000, 1)

    # 6 — Knowledge Retrieval
    _progress(6, "Agent 6/7 — Knowledge Retrieval (RAG)")
    t0 = time.time()
    if enable_rag:
        rag = run_knowledge_retrieval(result["corrected_text"], result["script"])
        result["validated_text"] = rag["validated_text"]
        result["rag_context"] = rag["context"]
    else:
        result["validated_text"] = result["corrected_text"]
        result["rag_context"] = ""
    result["agent_timings"]["KnowledgeRetrieval"] = round((time.time() - t0) * 1000, 1)

    # 7 — Output Formatting (annotated image)
    _progress(7, "Agent 7/7 — Output Formatting")
    t0 = time.time()
    annotated = create_annotated_image(working_bytes, regions)
    result["annotated_image_bytes"] = annotated
    result["final_text"] = result["validated_text"] or result["corrected_text"] or result["raw_text"]
    script_conf = result["script_confidence"] or 0.0
    ocr_conf = result["ocr_confidence"] or 0.0
    result["overall_confidence"] = round(script_conf * 0.35 + ocr_conf * 0.65, 4)
    result["agent_timings"]["OutputFormatting"] = round((time.time() - t0) * 1000, 1)

    return result
