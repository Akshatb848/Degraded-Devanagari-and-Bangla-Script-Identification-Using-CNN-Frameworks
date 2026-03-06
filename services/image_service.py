"""
Image service for preprocessing and quality assessment.
Wraps image restoration logic for use outside the agent pipeline.
"""
from typing import Dict, Any
import structlog

logger = structlog.get_logger()


class ImageRestorationService:
    """Service for image enhancement operations."""

    async def restore(
        self,
        image_bytes: bytes,
        enhance_resolution: bool = True,
        denoise: bool = True,
        deskew: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply image restoration pipeline.

        Returns:
            {
                restored_bytes: bytes,
                enhancements: list[str],
                quality_before: float,
                quality_after: float,
            }
        """
        import asyncio
        from agents.image_restoration_agent import (
            _bytes_to_cv2,
            _cv2_to_bytes,
            _compute_quality_score,
            _apply_clahe,
            _apply_super_resolution,
            _deskew,
        )
        import cv2

        def _process():
            np_img = _bytes_to_cv2(image_bytes)
            quality_before = _compute_quality_score(np_img)
            enhancements = []

            if deskew:
                result = _deskew(np_img)
                if result is not None:
                    np_img = result
                    enhancements.append("deskew")

            if denoise:
                if len(np_img.shape) == 3:
                    np_img_d = cv2.fastNlMeansDenoisingColored(np_img, None, 10, 10, 7, 21)
                else:
                    np_img_d = cv2.fastNlMeansDenoising(np_img, None, 10, 7, 21)
                np_img = np_img_d
                enhancements.append("denoising")

            np_img = _apply_clahe(np_img)
            enhancements.append("clahe_contrast")

            if enhance_resolution:
                np_img, method = _apply_super_resolution(np_img)
                enhancements.append(method)

            quality_after = _compute_quality_score(np_img)
            restored_bytes = _cv2_to_bytes(np_img)

            return {
                "restored_bytes": restored_bytes,
                "enhancements": enhancements,
                "quality_before": quality_before,
                "quality_after": quality_after,
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process)
