"""
Bird Motion Detector  v15.0
═══════════════════════════════════════════════════════════════════════════════
Changes from v14.0:
  • bad_frame_reject via FrameAnalyzer
  • ROI normalised dict accepted
  • Improved logging
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

DETECT_WIDTH      = 640
PIXEL_THRESHOLD   = 30
MIN_CONTOUR_AREA  = 250
REF_BUFFER_LEN    = 6
FLICKER_RATIO     = 0.40
EXPOSURE_STEP_THR = 18
CONSISTENCY_WINDOW = 3
CONSECUTIVE_MIN    = 2


class MotionDetector:
    """
    OpenCL-accelerated adjacent-frame diff detector with ROI, MOG2,
    flicker rejection, temporal consistency gating, and bad-frame removal.
    """

    def __init__(self, config: dict):
        sens = max(1, min(10, int(config.get("sensitivity", 5))))
        self.motion_threshold  = int(380 * (1.0 + (sens - 1) * 1.1))
        self.white_thr         = int(config.get("white_threshold",   225))
        self.black_thr         = int(config.get("black_threshold",    35))
        self.pad_sec           = float(config.get("segment_padding",  0.3))
        self.frame_skip        = max(1, int(config.get("frame_skip",   2)))
        self.min_motion_s      = float(config.get("min_motion_duration", 0.4))
        self.merge_gap_s       = float(config.get("merge_gap",         0.8))
        self.bad_frame_reject  = bool(config.get("bad_frame_reject",  True))
        self.roi: Optional[dict] = config.get("roi")
        self.use_mog2          = bool(config.get("use_mog2",         False))
        self.mog2_lr           = float(config.get("mog2_learning_rate", 0.005))
        self._cancel: Optional[threading.Event] = None
        self._mog2:   Optional[cv2.BackgroundSubtractorMOG2] = None

        logger.info(
            f"MotionDetector v15.0 | sens={sens} threshold={self.motion_threshold}px "
            f"pad={self.pad_sec}s skip={self.frame_skip} "
            f"roi={'yes' if self.roi else 'no'} mog2={self.use_mog2}"
        )

    def set_cancel_flag(self, flag: threading.Event) -> None:
        self._cancel = flag

    def detect_motion(
        self,
        video_path:        str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[Tuple[float, float]], dict]:
        import time
        logger.info(f"[DETECT] Starting → {video_path}")
        t0 = time.time()

        raw_indices, meta = self._scan(video_path, progress_callback)
        fps          = meta["fps"]
        total_frames = meta["total_frames"]
        duration     = meta["duration"]

        segments   = self._build_segments(
            sorted(set(raw_indices)), fps, total_frames,
            self.min_motion_s, self.pad_sec, self.merge_gap_s,
        )
        motion_dur = sum(e - s for s, e in segments)
        elapsed    = time.time() - t0

        logger.info(
            f"[DETECT] Complete | segments={len(segments)} "
            f"motion={motion_dur:.1f}s ({motion_dur/max(duration,1)*100:.1f}%) "
            f"elapsed={elapsed:.1f}s"
        )
        return segments, {
            **meta,
            "motion_segments":   len(segments),
            "motion_duration":   motion_dur,
            "detection_method":  "v15.0",
            "detection_elapsed": elapsed,
        }

    def _scan(self, path, cb=None):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        det_h  = max(1, int(orig_h * DETECT_WIDTH / max(orig_w, 1)))

        roi_mask = self._build_roi_mask(det_h, self.roi)

        if self.use_mog2:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=False)
        else:
            self._mog2 = None

        ref_buf:       deque     = deque(maxlen=REF_BUFFER_LEN)
        recent_scores: deque     = deque(maxlen=CONSISTENCY_WINDOW)
        motion_indices: List[int] = []
        frame_no = -1; processed = 0; suppressed = 0

        while True:
            if self._cancel and self._cancel.is_set():
                logger.info("[DETECT] Cancelled"); break

            ok, frame = cap.read()
            if not ok: break
            frame_no += 1

            if frame_no % self.frame_skip != 0:
                continue
            if cb and processed % 300 == 0:
                cb(min(99.0, frame_no / max(total, 1) * 100))

            gray = self._to_gray(frame, det_h)
            if gray is None:
                processed += 1; continue

            if roi_mask is not None:
                gray = cv2.bitwise_and(gray, roi_mask)

            if len(ref_buf) < REF_BUFFER_LEN:
                ref_buf.append(gray); processed += 1; continue

            ref_gray = ref_buf[0]
            cur_mean = float(np.mean(gray))
            ref_mean = float(np.mean(ref_gray))
            if ref_mean > 0 and abs(cur_mean - ref_mean) > EXPOSURE_STEP_THR:
                scale    = cur_mean / ref_mean
                ref_gray = np.clip(ref_gray.astype(np.float32) * scale, 0, 255).astype(np.uint8)
                suppressed += 1

            cur_u  = cv2.UMat(gray)
            ref_u  = cv2.UMat(ref_gray)
            diff_u = cv2.absdiff(cur_u, ref_u)
            diff_np = diff_u.get()

            mean_d = float(np.mean(diff_np))
            if mean_d < 1.5:
                ref_buf.append(gray); recent_scores.append(0); processed += 1; continue

            std_d = float(np.std(diff_np))
            ratio = std_d / mean_d
            if ratio < FLICKER_RATIO:
                recent_scores.append(0); suppressed += 1; processed += 1; continue

            _, mask_u = cv2.threshold(diff_u, PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
            score     = cv2.countNonZero(mask_u)

            is_motion = False
            if score >= self.motion_threshold:
                mask_np    = mask_u.get()
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big = sum(1 for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA)
                if big > 0:
                    if self._mog2 is not None:
                        fg = self._mog2.apply(gray, learningRate=self.mog2_lr)
                        if int(np.count_nonzero(fg)) >= self.motion_threshold // 2:
                            is_motion = True
                    else:
                        is_motion = True

            if not is_motion and self._mog2 is not None:
                self._mog2.apply(gray, learningRate=self.mog2_lr)

            recent_scores.append(1 if is_motion else 0)
            if sum(recent_scores) >= CONSECUTIVE_MIN:
                motion_indices.append(frame_no)

            ref_buf.append(gray)
            processed += 1

        cap.release()
        logger.info(
            f"[DETECT] Scan: motion_frames={len(motion_indices)} "
            f"processed={processed} suppressed={suppressed}"
        )
        return motion_indices, {
            "fps": fps, "total_frames": total, "duration": total / fps,
            "orig_w": orig_w, "orig_h": orig_h,
            "processed": processed, "suppressed": suppressed,
        }

    def _to_gray(self, frame, det_h):
        small = cv2.resize(frame, (DETECT_WIDTH, det_h), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean  = float(np.mean(gray))
        if mean > self.white_thr or mean < self.black_thr:
            return None
        return gray

    @staticmethod
    def _build_roi_mask(det_h, roi):
        if not roi: return None
        mask = np.zeros((det_h, DETECT_WIDTH), dtype=np.uint8)
        x = max(0, int(roi.get("x", 0)   * DETECT_WIDTH))
        y = max(0, int(roi.get("y", 0)   * det_h))
        w = max(1, int(roi.get("w", 1.0) * DETECT_WIDTH))
        h = max(1, int(roi.get("h", 1.0) * det_h))
        mask[y:y+h, x:x+w] = 255
        logger.info(f"[DETECT] ROI mask: x={x} y={y} w={w} h={h}")
        return mask

    @staticmethod
    def _build_segments(indices, fps, total_frames, min_sec, padding_sec, merge_gap_sec):
        if not indices: return []
        gap_f   = fps * merge_gap_sec
        pad_f   = int(padding_sec * fps)
        total_t = total_frames / fps

        raw = []
        s = e = indices[0]
        for f in indices[1:]:
            if f - e <= gap_f: e = f
            else: raw.append((s, e)); s = e = f
        raw.append((s, e))

        padded = []
        for s, e in raw:
            ts = max(0.0,     (s - pad_f) / fps)
            te = min(total_t, (e + pad_f) / fps)
            if te - ts >= min_sec:
                padded.append((ts, te))
        if not padded: return []

        merged = [list(padded[0])]
        for ts, te in padded[1:]:
            if ts <= merged[-1][1] + 0.1: merged[-1][1] = max(merged[-1][1], te)
            else: merged.append([ts, te])

        return [(float(s), float(e)) for s, e in merged]
