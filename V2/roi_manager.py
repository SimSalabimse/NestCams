"""
roi_manager.py — Interactive and persistent ROI (Region of Interest) handling.
"""

import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ROI = Tuple[int, int, int, int]   # (x, y, w, h) in pixels


class ROIManager:
    """
    Manages rectangular ROI selection for motion detection.
    Stored as normalised fractions for resolution-independence.
    """

    def __init__(self, config_path: str = "roi_config.json"):
        self.config_path = config_path

    def select_roi_interactive(self, first_frame: np.ndarray) -> Optional[ROI]:
        """Open an OpenCV window for headless/CLI ROI drawing. Returns (x,y,w,h) or None."""
        WINDOW = "Draw ROI — drag rectangle, ENTER to confirm, ESC to cancel"
        clone  = first_frame.copy()
        state  = {"start": None, "end": None, "drawing": False, "done": False}

        def _mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["start"] = (x, y); state["drawing"] = True; state["done"] = False
            elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
                state["end"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state["end"] = (x, y); state["drawing"] = False; state["done"] = True

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, _mouse)

        while True:
            display = clone.copy()
            if state["start"] and state["end"]:
                cv2.rectangle(display, state["start"], state["end"], (124, 58, 237), 2)
            cv2.imshow(WINDOW, display)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and state["done"]:
                break
            if k == 27:
                cv2.destroyWindow(WINDOW)
                return None

        cv2.destroyWindow(WINDOW)
        if not (state["start"] and state["end"]):
            return None

        x0, y0 = state["start"]
        x1, y1 = state["end"]
        x, y   = min(x0, x1), min(y0, y1)
        w, h   = abs(x1 - x0), abs(y1 - y0)
        if w < 5 or h < 5:
            logger.warning("[ROI] Selection too small")
            return None
        logger.info(f"[ROI] Selected: x={x} y={y} w={w} h={h}")
        return (x, y, w, h)

    def normalise_roi(self, roi: ROI, frame_w: int, frame_h: int) -> dict:
        x, y, w, h = roi
        return {
            "x": round(x / frame_w, 4), "y": round(y / frame_h, 4),
            "w": round(w / frame_w, 4), "h": round(h / frame_h, 4),
        }

    def denormalise_roi(self, norm: dict, frame_w: int, frame_h: int) -> ROI:
        return (
            int(norm["x"] * frame_w), int(norm["y"] * frame_h),
            int(norm["w"] * frame_w), int(norm["h"] * frame_h),
        )

    def draw_roi_preview(
        self, frame: np.ndarray, roi: ROI,
        color: Tuple[int, int, int] = (124, 58, 237),
    ) -> np.ndarray:
        out = frame.copy()
        x, y, w, h = roi
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, "ROI", (x + 4, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return out

    def save_roi(self, roi_norm: dict) -> None:
        try:
            data = {}
            if Path(self.config_path).exists():
                with open(self.config_path) as fh:
                    data = json.load(fh)
            data["roi"] = roi_norm
            with open(self.config_path, "w") as fh:
                json.dump(data, fh, indent=2)
            logger.info(f"[ROI] Saved: {roi_norm}")
        except Exception as exc:
            logger.error(f"[ROI] Save failed: {exc}")

    def load_roi(self) -> Optional[dict]:
        try:
            if not Path(self.config_path).exists():
                return None
            with open(self.config_path) as fh:
                data = json.load(fh)
            return data.get("roi")
        except Exception as exc:
            logger.warning(f"[ROI] Load failed: {exc}")
            return None

    def build_mask(self, det_h: int, det_w: int, roi_norm: dict) -> np.ndarray:
        mask = np.zeros((det_h, det_w), dtype=np.uint8)
        x = max(0, int(roi_norm.get("x", 0) * det_w))
        y = max(0, int(roi_norm.get("y", 0) * det_h))
        w = max(1, int(roi_norm.get("w", 1) * det_w))
        h = max(1, int(roi_norm.get("h", 1) * det_h))
        mask[y:y + h, x:x + w] = 255
        return mask
