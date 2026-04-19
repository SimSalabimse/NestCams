"""
auto_tuner.py — Auto-tunes motion detection sensitivity from a short sample pass.
"""

import logging
import time
from typing import Optional, Callable

import cv2
import numpy as np

from motion_detector import MotionDetector

logger = logging.getLogger(__name__)


class AutoTuner:
    """
    Runs a quick multi-pass analysis on a short sample of the input video
    to suggest optimal motion detection settings.

    Strategy:
      1. Extract a 3-minute window from mid-video.
      2. Run detection at sensitivity 3, 5, 7 (and 2, 8 if needed).
      3. Score each result (target: 2–25% motion density, ideal ~10%).
      4. Return the config that scores closest to ideal.
    """

    TARGET_DENSITY_LOW   = 0.02
    TARGET_DENSITY_HIGH  = 0.25
    TARGET_DENSITY_IDEAL = 0.10
    SAMPLE_DURATION_SEC  = 180

    def __init__(self, base_config: Optional[dict] = None):
        self.base_config = base_config or {}

    def suggest_optimal_settings(
        self,
        sample_video:      str,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        logger.info(f"[AUTOTUNE] Starting: {sample_video}")
        t0 = time.time()

        sample_path = self._extract_sample(sample_video)
        if sample_path is None:
            logger.warning("[AUTOTUNE] Sample extraction failed — using defaults")
            return self._default_config()

        results = []
        for sens in [3, 5, 7, 2, 8]:
            if progress_callback:
                progress_callback((len(results) / 5) * 100)
            cfg = {**self.base_config, "sensitivity": sens, "frame_skip": 3}
            det = MotionDetector(cfg)
            try:
                segs, stats = det.detect_motion(sample_path)
                duration    = stats.get("duration", self.SAMPLE_DURATION_SEC)
                motion_dur  = stats.get("motion_duration", 0)
                density     = motion_dur / max(duration, 1)
                score       = self._density_score(density)
                results.append({"sensitivity": sens, "density": density,
                                 "score": score, "stats": stats})
                logger.info(f"[AUTOTUNE] sens={sens} density={density*100:.1f}% score={score:.3f}")
                if score > 0.9:
                    break
            except Exception as exc:
                logger.warning(f"[AUTOTUNE] sens={sens} failed: {exc}")

        try:
            import os; os.remove(sample_path)
        except OSError:
            pass

        if not results:
            return self._default_config()

        best              = max(results, key=lambda r: r["score"])
        sensitivity       = best["sensitivity"]
        brightness_stats  = self._probe_brightness(sample_video)

        suggested = {
            **self.base_config,
            "sensitivity":    sensitivity,
            "white_threshold": min(225, int(brightness_stats["p99"] + 10)),
            "black_threshold": max(20,  int(brightness_stats["p01"] - 5)),
            "frame_skip":      self._recommend_frame_skip(best["stats"].get("duration", 180)),
            "deflicker_size":  5 if brightness_stats["flicker_score"] > 0.3 else 3,
        }

        logger.info(f"[AUTOTUNE] Done in {time.time()-t0:.1f}s | best_sens={sensitivity}")
        return suggested

    def _extract_sample(self, video_path: str) -> Optional[str]:
        import subprocess, tempfile, os
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=15,
            )
            total = float(r.stdout.strip() or "0")
            start = max(0.0, total / 2 - self.SAMPLE_DURATION_SEC / 2)
            dur   = min(self.SAMPLE_DURATION_SEC, total)
            fd, out = tempfile.mkstemp(suffix="_autotune.mp4")
            os.close(fd)
            rc = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-i", video_path,
                 "-t", str(dur), "-c", "copy", out],
                capture_output=True, timeout=120,
            )
            if rc.returncode == 0 and os.path.getsize(out) > 10000:
                return out
            os.remove(out)
            return None
        except Exception as exc:
            logger.warning(f"[AUTOTUNE] Sample extraction: {exc}")
            return None

    def _probe_brightness(self, video_path: str) -> dict:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        means: list = []
        step  = max(1, total // 60)
        for i in range(0, min(total, step * 60), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                means.append(float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))))
        cap.release()
        if not means:
            return {"p01": 30, "p99": 225, "flicker_score": 0.0}
        arr   = np.array(means)
        diffs = np.abs(np.diff(arr))
        return {
            "p01":           float(np.percentile(arr, 1)),
            "p99":           float(np.percentile(arr, 99)),
            "mean":          float(np.mean(arr)),
            "flicker_score": float(np.mean(diffs > 15)),
        }

    def _density_score(self, density: float) -> float:
        if density < self.TARGET_DENSITY_LOW:
            return density / self.TARGET_DENSITY_LOW
        if density > self.TARGET_DENSITY_HIGH:
            return max(0.0, 1.0 - (density - self.TARGET_DENSITY_HIGH) / 0.5)
        dist = abs(density - self.TARGET_DENSITY_IDEAL)
        return max(0.5, 1.0 - dist / self.TARGET_DENSITY_IDEAL)

    @staticmethod
    def _recommend_frame_skip(duration_sec: float) -> int:
        if duration_sec < 1800: return 2
        if duration_sec < 7200: return 3
        return 4

    @staticmethod
    def _default_config() -> dict:
        return {"sensitivity": 5, "white_threshold": 225,
                "black_threshold": 35, "frame_skip": 2, "deflicker_size": 5}
