"""
core — modular pipeline components for AIOCR
"""

from .preprocessing import preprocess, PreprocessResult
from .classifier    import classify_script, get_classifier
from .ocr_engine    import run_ocr, OCRResult
from .llm_corrector import correct_text, CorrectionResult
from .logger        import RequestLog, StepMetric

__all__ = [
    "preprocess", "PreprocessResult",
    "classify_script", "get_classifier",
    "run_ocr", "OCRResult",
    "correct_text", "CorrectionResult",
    "RequestLog", "StepMetric",
]
