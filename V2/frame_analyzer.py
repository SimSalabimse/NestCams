"""
frame_analyzer.py — Standalone bad-frame detector and quality scorer.
"""

import cv2
import numpy as np
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class FrameAnalyzer:
    """
    Detects overexposed, underexposed, uniformly coloured, or corrupted frames.
    Also provides a quality score (0.0–1.0) and basic inpainting repair.
    """

    def __init__(self, white_thresh: int = 230, black_thresh: int = 30, min_std: float = 8.0):
        self.white_thresh = white_thresh
        self.black_thresh = black_thresh
        self.min_std      = min_std

    def is_bad_frame(
        self,
        frame:        np.ndarray,
        white_thresh: Optional[int]   = None,
        black_thresh: Optional[int]   = None,
        min_std:      Optional[float] = None,
    ) -> bool:
        if frame is None or frame.size == 0:
            return True
        if np.isnan(frame).any():
            return True
        wt  = white_thresh if white_thresh is not None else self.white_thresh
        bt  = black_thresh if black_thresh is not None else self.black_thresh
        mst = min_std      if min_std      is not None else self.min_std
        gray = self._to_gray(frame)
        mean = float(np.mean(gray))
        std  = float(np.std(gray))
        if mean >= wt:
            logger.debug(f"[FA] Bad frame: overexposed (mean={mean:.1f})")
            return True
        if mean <= bt:
            logger.debug(f"[FA] Bad frame: underexposed (mean={mean:.1f})")
            return True
        if std < mst:
            logger.debug(f"[FA] Bad frame: too uniform (std={std:.1f})")
            return True
        return False

    def score_frame_quality(self, frame: np.ndarray) -> float:
        if self.is_bad_frame(frame):
            return 0.0
        gray    = self._to_gray(frame)
        mean    = float(np.mean(gray))
        std     = float(np.std(gray))
        bscore  = max(0.0, 1.0 - abs(mean - 125) / 125) * 0.4
        cscore  = min(1.0, std / 60.0) * 0.3
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sscore  = min(1.0, lap_var / 500.0) * 0.3
        return round(min(1.0, bscore + cscore + sscore), 3)

    def repair_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray = self._to_gray(frame)
            lo   = np.percentile(gray, 0.5)
            hi   = np.percentile(gray, 99.5)
            mask = ((gray < lo) | (gray > hi)).astype(np.uint8) * 255
            if mask.sum() == 0:
                return frame
            return cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        except Exception as exc:
            logger.debug(f"[FA] repair_frame failed: {exc}")
            return frame

    def find_best_frame(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        scored = [(self.score_frame_quality(f), f) for f in frames if f is not None]
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_frame = scored[0]
        return best_frame if best_score > 0.1 else None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
