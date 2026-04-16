"""
Bird Motion Detector  v14.0
═══════════════════════════════════════════════════════════════════════════════

ALGORITHM OVERVIEW
──────────────────
1. Downsample each frame to DETECT_WIDTH for speed.
2. Skip over/under-exposed frames (black/white threshold).
3. Optional ROI mask: zero-out pixels outside user-defined rectangle so that
   leaves, shadows, and camera movement outside the box are ignored entirely.
4. Maintain a circular reference buffer (default 6 frames).  Diff against the
   OLDEST frame in the buffer — this creates an adaptive ~0.4 s temporal gap
   that catches slow-moving birds without being confused by gradual light drift.
5. Exposure-step correction: if the camera auto-exposed or the IR-cut filter
   switched, global brightness jumps.  We scale the reference frame to match
   current brightness BEFORE computing the diff so the jump is absorbed.
6. Spatial-uniformity flicker rejection: real motion creates a LOCALISED diff
   patch (high std/mean ratio).  A lighting flicker creates a UNIFORM diff
   (low std/mean ratio).  Frames below FLICKER_RATIO are suppressed.
7. Pixel-count threshold + contour size filter: require at least one blob
   larger than MIN_CONTOUR_AREA to avoid random sensor noise.
8. Optional MOG2 background subtractor as a secondary confirmation gate.
9. Temporal consistency gate: require CONSECUTIVE_MIN positives in the last
   CONSISTENCY_WINDOW frames to filter out single-frame hits.
10. Build time segments from confirmed motion-frame indices.
"""

import cv2
import numpy as np
import logging
import threading
from collections import deque
from typing import List, Tuple, Optional, Callable

cv2.ocl.setUseOpenCL(True)
cv2.setNumThreads(0)

logger = logging.getLogger(__name__)

# ── Tunable constants (can be overridden via config) ──────────────────────────
DETECT_WIDTH        = 640
PIXEL_THRESHOLD     = 30
MIN_CONTOUR_AREA    = 250
REF_BUFFER_LEN      = 6
FLICKER_RATIO       = 0.40
EXPOSURE_STEP_THR   = 18
CONSISTENCY_WINDOW  = 3
CONSECUTIVE_MIN     = 2


class MotionDetector:
    """
    OpenCL-accelerated adjacent-frame diff detector with ROI, MOG2 option,
    flicker rejection, and temporal consistency gating.
    """

    def __init__(self, config: dict):
        sens = max(1, min(10, int(config.get("sensitivity", 5))))
        # sens 1 → ~380 px  sens 5 → ~2280 px  sens 10 → ~7980 px
        self.motion_threshold = int(380 * (1.0 + (sens - 1) * 1.1))

        self.white_thr    = int(config.get("white_threshold",  225))
        self.black_thr    = int(config.get("black_threshold",   35))
        self.pad_sec      = float(config.get("segment_padding", 0.3))
        self.frame_skip   = max(1, int(config.get("frame_skip",   2)))
        self.min_motion_s = float(config.get("min_motion_duration", 0.4))
        self.merge_gap_s  = float(config.get("merge_gap", 0.8))

        # ROI: dict with keys x, y, w, h (all 0–1 normalised fractions)
        # e.g. {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}
        self.roi: Optional[dict] = config.get("roi")

        # MOG2 options
        self.use_mog2   = bool(config.get("use_mog2", False))
        self.mog2_lr    = float(config.get("mog2_learning_rate", 0.005))

        self._cancel: Optional[threading.Event] = None
        self._mog2: Optional[cv2.BackgroundSubtractorMOG2] = None

        logger.info(
            "MotionDetector v14.0 | "
            f"sens={sens} threshold={self.motion_threshold}px "
            f"pad={self.pad_sec}s skip={self.frame_skip} "
            f"roi={'yes' if self.roi else 'no'} "
            f"mog2={self.use_mog2} "
            f"flicker_ratio={FLICKER_RATIO} "
            f"consistency={CONSECUTIVE_MIN}/{CONSISTENCY_WINDOW}"
        )

    def set_cancel_flag(self, flag: threading.Event) -> None:
        self._cancel = flag

    # ─────────────────────────────────────────────────── public interface ──

    def detect_motion(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[Tuple[float, float]], dict]:
        """
        Run detection on *video_path*.
        Returns (segments, stats_dict).
        segments = [(start_sec, end_sec), …]  sorted, non-overlapping.
        """
        logger.info(f"[DETECT] Starting → {video_path}")
        t0 = __import__("time").time()

        raw_indices, meta = self._scan(video_path, progress_callback)

        fps          = meta["fps"]
        total_frames = meta["total_frames"]
        duration     = meta["duration"]

        segments = self._build_segments(
            sorted(set(raw_indices)), fps, total_frames,
            self.min_motion_s, self.pad_sec, self.merge_gap_s,
        )

        motion_dur = sum(e - s for s, e in segments)
        elapsed = __import__("time").time() - t0

        logger.info(
            f"[DETECT] Complete | "
            f"segments={len(segments)} "
            f"motion={motion_dur:.1f}s "
            f"({motion_dur/max(duration,1)*100:.1f}%) "
            f"elapsed={elapsed:.1f}s "
            f"suppressed={meta.get('suppressed',0)}"
        )

        return segments, {
            **meta,
            "motion_segments":  len(segments),
            "motion_duration":  motion_dur,
            "detection_method": "v14.0",
            "detection_elapsed": elapsed,
        }

    # ───────────────────────────────────────────────────── scan loop ──────

    def _scan(
        self,
        path: str,
        cb: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[int], dict]:

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        det_h  = max(1, int(orig_h * DETECT_WIDTH / max(orig_w, 1)))

        # Build ROI mask (at detection resolution)
        roi_mask = self._build_roi_mask(det_h, self.roi)

        # MOG2 initialised fresh for each video
        if self.use_mog2:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=False
            )
        else:
            self._mog2 = None

        ref_buf: deque          = deque(maxlen=REF_BUFFER_LEN)
        recent_scores: deque    = deque(maxlen=CONSISTENCY_WINDOW)
        motion_indices: List[int] = []

        frame_no   = -1
        processed  = 0
        suppressed = 0

        while True:
            if self._cancel and self._cancel.is_set():
                logger.info("[DETECT] Cancelled by user")
                break

            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1

            if frame_no % self.frame_skip != 0:
                continue

            if cb and processed % 300 == 0:
                cb(min(99.0, frame_no / max(total, 1) * 100))

            gray = self._to_gray(frame, det_h)
            if gray is None:
                # Bad (over/under-exposed) frame — do not add to ref buffer
                processed += 1
                logger.debug(f"[DETECT] Frame {frame_no}: bad brightness, skipped")
                continue

            # Apply ROI mask
            if roi_mask is not None:
                gray = cv2.bitwise_and(gray, roi_mask)

            # ── Need full buffer before first comparison ───────────────────
            if len(ref_buf) < REF_BUFFER_LEN:
                ref_buf.append(gray)
                processed += 1
                continue

            ref_gray = ref_buf[0]   # oldest frame in buffer

            # ── Exposure-step correction ──────────────────────────────────
            cur_mean = float(np.mean(gray))
            ref_mean = float(np.mean(ref_gray))
            corrected = False
            if ref_mean > 0 and abs(cur_mean - ref_mean) > EXPOSURE_STEP_THR:
                scale    = cur_mean / ref_mean
                ref_gray = np.clip(
                    ref_gray.astype(np.float32) * scale, 0, 255
                ).astype(np.uint8)
                corrected = True
                suppressed += 1
                logger.debug(
                    f"[DETECT] Frame {frame_no}: exposure step "
                    f"Δ={cur_mean - ref_mean:.1f} → rescaling reference"
                )

            # ── Diff on GPU ────────────────────────────────────────────────
            cur_u  = cv2.UMat(gray)
            ref_u  = cv2.UMat(ref_gray)
            diff_u = cv2.absdiff(cur_u, ref_u)
            diff_np = diff_u.get()

            # ── Spatial uniformity flicker check ──────────────────────────
            mean_d = float(np.mean(diff_np))
            if mean_d < 1.5:
                ref_buf.append(gray)
                recent_scores.append(0)
                processed += 1
                continue

            std_d = float(np.std(diff_np))
            ratio = std_d / mean_d
            if ratio < FLICKER_RATIO:
                # Uniform change → flicker / IR-cut / global glare
                # Do NOT advance ref buffer so next event still compares clean
                recent_scores.append(0)
                suppressed += 1
                logger.debug(
                    f"[DETECT] Frame {frame_no}: flicker rejected "
                    f"(ratio={ratio:.3f} < {FLICKER_RATIO})"
                )
                processed += 1
                continue

            # ── Pixel-count threshold ──────────────────────────────────────
            _, mask_u = cv2.threshold(diff_u, PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
            score = cv2.countNonZero(mask_u)

            is_motion = False
            if score >= self.motion_threshold:
                mask_np = mask_u.get()
                contours, _ = cv2.findContours(
                    mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                big = sum(1 for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA)
                if big > 0:
                    # ── Optional MOG2 secondary check ─────────────────────
                    if self._mog2 is not None:
                        fg = self._mog2.apply(gray, learningRate=self.mog2_lr)
                        mog2_score = int(np.count_nonzero(fg))
                        if mog2_score >= self.motion_threshold // 2:
                            is_motion = True
                        else:
                            logger.debug(
                                f"[DETECT] Frame {frame_no}: MOG2 rejected "
                                f"(mog2={mog2_score}, need≥{self.motion_threshold//2})"
                            )
                    else:
                        is_motion = True

            if not is_motion and self._mog2 is not None:
                # Still call apply() to keep MOG2 model updated
                self._mog2.apply(gray, learningRate=self.mog2_lr)

            recent_scores.append(1 if is_motion else 0)

            # ── Temporal consistency gate ──────────────────────────────────
            if sum(recent_scores) >= CONSECUTIVE_MIN:
                motion_indices.append(frame_no)
                logger.debug(
                    f"[DETECT] Frame {frame_no}: MOTION confirmed "
                    f"score={score} ratio={ratio:.2f}"
                )

            # Always advance reference buffer
            ref_buf.append(gray)
            processed += 1

        cap.release()

        logger.info(
            f"[DETECT] Scan complete: "
            f"motion_frames={len(motion_indices)} "
            f"processed={processed} suppressed={suppressed}"
        )
        return motion_indices, {
            "fps": fps, "total_frames": total, "duration": total / fps,
            "orig_w": orig_w, "orig_h": orig_h,
            "processed": processed, "suppressed": suppressed,
        }

    # ─────────────────────────────────────────────────────── helpers ──────

    def _to_gray(self, frame, det_h: int) -> Optional[np.ndarray]:
        small = cv2.resize(frame, (DETECT_WIDTH, det_h), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean  = float(np.mean(gray))
        if mean > self.white_thr or mean < self.black_thr:
            return None
        return gray

    @staticmethod
    def _build_roi_mask(
        det_h: int,
        roi: Optional[dict],
    ) -> Optional[np.ndarray]:
        """
        Build a binary mask at detection resolution from a normalised ROI dict.
        roi = {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}  (fractions 0–1)
        Returns None if no ROI configured.
        """
        if not roi:
            return None
        mask = np.zeros((det_h, DETECT_WIDTH), dtype=np.uint8)
        x  = max(0, int(roi.get("x", 0)   * DETECT_WIDTH))
        y  = max(0, int(roi.get("y", 0)   * det_h))
        w  = max(1, int(roi.get("w", 1.0) * DETECT_WIDTH))
        h  = max(1, int(roi.get("h", 1.0) * det_h))
        mask[y:y+h, x:x+w] = 255
        logger.info(
            f"[DETECT] ROI mask built: "
            f"x={x} y={y} w={w} h={h} "
            f"(detect res {DETECT_WIDTH}×{det_h})"
        )
        return mask

    @staticmethod
    def _build_segments(
        indices: List[int],
        fps: float, total_frames: int,
        min_sec: float, padding_sec: float, merge_gap_sec: float,
    ) -> List[Tuple[float, float]]:
        if not indices:
            return []

        gap_f   = fps * merge_gap_sec
        pad_f   = int(padding_sec * fps)
        total_t = total_frames / fps

        raw = []
        s = e = indices[0]
        for f in indices[1:]:
            if f - e <= gap_f:
                e = f
            else:
                raw.append((s, e)); s = e = f
        raw.append((s, e))

        padded = []
        for s, e in raw:
            ts = max(0.0,     (s - pad_f) / fps)
            te = min(total_t, (e + pad_f) / fps)
            if te - ts >= min_sec:
                padded.append((ts, te))
        if not padded:
            return []

        merged = [list(padded[0])]
        for ts, te in padded[1:]:
            if ts <= merged[-1][1] + 0.1:
                merged[-1][1] = max(merged[-1][1], te)
            else:
                merged.append([ts, te])

        return [(float(s), float(e)) for s, e in merged]
