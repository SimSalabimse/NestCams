"""
thumbnail_generator.py — Auto-generates multiple thumbnail variants per video.
"""

import cv2
import logging
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """
    Generates up to 3 thumbnail variants at 1280×720 JPEG:
      1. ActionFreeze     — frame from the highest-activity moment
      2. SplitBeforeAfter — side-by-side first vs best frame
      3. TextOverlay      — best frame with a bold text banner
    """

    SIZE = (1280, 720)

    def __init__(self, logo_path: Optional[str] = None):
        self.logo_path  = logo_path
        self._logo_img: Optional[np.ndarray] = None
        if logo_path and os.path.exists(logo_path):
            self._logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    def generate_all(
        self, video_path: str, segments: Optional[list] = None,
        output_dir: Optional[str] = None, text: str = "",
    ) -> List[str]:
        if output_dir is None:
            output_dir = str(Path(video_path).parent)
        os.makedirs(output_dir, exist_ok=True)

        stem   = Path(video_path).stem
        frames = self._sample_frames(video_path, segments)
        if not frames:
            return []

        from frame_analyzer import FrameAnalyzer
        fa    = FrameAnalyzer()
        best  = fa.find_best_frame(frames) or frames[0]
        first = frames[0]
        outputs: List[str] = []

        p1 = os.path.join(output_dir, f"{stem}_thumb_action.jpg")
        self._save_action(best, p1, text)
        outputs.append(p1)

        p2 = os.path.join(output_dir, f"{stem}_thumb_split.jpg")
        self._save_split(first, best, p2)
        outputs.append(p2)

        p3 = os.path.join(output_dir, f"{stem}_thumb_text.jpg")
        self._save_text_overlay(best, p3, text or "🐦 Watch What Happens!")
        outputs.append(p3)

        logger.info(f"[THUMB] {len(outputs)} thumbnails for {stem}")
        return outputs

    def _save_action(self, frame: np.ndarray, path: str, text: str = "") -> None:
        out = cv2.resize(frame, self.SIZE, interpolation=cv2.INTER_LANCZOS4)
        if self._logo_img is not None:
            out = self._overlay_logo(out)
        if text:
            out = self._draw_banner(out, text)
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def _save_split(self, left: np.ndarray, right: np.ndarray, path: str) -> None:
        hw = self.SIZE[0] // 2
        l  = cv2.resize(left,  (hw, self.SIZE[1]), interpolation=cv2.INTER_LANCZOS4)
        r  = cv2.resize(right, (hw, self.SIZE[1]), interpolation=cv2.INTER_LANCZOS4)
        out = np.hstack([l, r])
        cv2.line(out, (hw, 0), (hw, self.SIZE[1]), (255, 255, 255), 3)
        self._put_label(out, "BEFORE", (10, 40))
        self._put_label(out, "AFTER",  (hw + 10, 40))
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def _save_text_overlay(self, frame: np.ndarray, path: str, text: str) -> None:
        out = cv2.resize(frame, self.SIZE, interpolation=cv2.INTER_LANCZOS4)
        out = self._draw_banner(out, text, bottom=True)
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def _sample_frames(self, video_path: str, segments: Optional[list], n: int = 12) -> List[np.ndarray]:
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: List[np.ndarray] = []

        if segments:
            positions = [int(((s + e) / 2) * fps) for s, e in segments[:n]]
        else:
            positions = [int(total * i / n) for i in range(n)]

        for pos in positions[:n]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos))
            ok, f = cap.read()
            if ok: frames.append(f)
        cap.release()
        return frames

    def _draw_banner(self, img: np.ndarray, text: str, bottom: bool = False) -> np.ndarray:
        out   = img.copy()
        h, w  = out.shape[:2]
        bh    = 80
        y0    = h - bh if bottom else 0
        overlay = out.copy()
        cv2.rectangle(overlay, (0, y0), (w, y0 + bh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        cv2.putText(out, text, (20, y0 + 54), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _overlay_logo(self, img: np.ndarray, margin: int = 16) -> np.ndarray:
        if self._logo_img is None: return img
        logo    = self._logo_img
        lh, lw  = logo.shape[:2]
        max_w   = self.SIZE[0] // 5
        if lw > max_w:
            scale = max_w / lw
            logo  = cv2.resize(logo, (max_w, int(lh * scale)))
            lh, lw = logo.shape[:2]
        x1, y1 = img.shape[1] - lw - margin, img.shape[0] - lh - margin
        if logo.shape[2] == 4:
            alpha = logo[:, :, 3:] / 255.0
            roi   = img[y1:y1+lh, x1:x1+lw]
            img[y1:y1+lh, x1:x1+lw] = (roi * (1 - alpha) + logo[:, :, :3] * alpha).astype(np.uint8)
        else:
            img[y1:y1+lh, x1:x1+lw] = logo[:, :, :3]
        return img

    @staticmethod
    def _put_label(img: np.ndarray, text: str, pos: Tuple[int, int]) -> None:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
