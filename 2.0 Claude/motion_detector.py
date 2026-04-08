"""
Motion Detection Module - OpenCL Accelerated
Uses OpenCV UMat API for GPU-accelerated detection via OpenCL.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Callable, Optional

# Enable OpenCL
cv2.ocl.setUseOpenCL(True)
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "NVIDIA:GPU")

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self, config: dict):
        self.config = config
        self.sensitivity = config.get("sensitivity", 5)
        self.min_motion_duration = config.get("min_motion_duration", 0.5)
        self.segment_padding = config.get("segment_padding", 1.0)
        self.frame_skip = config.get("frame_skip", 2)
        self.detection_scale = config.get("detection_scale", 640)

        self.motion_threshold = int(8000 * (self.sensitivity / 5.0))
        self.bg_learn_rate = config.get("bg_learn_rate", 0.02)

        self.use_opencl = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
        if self.use_opencl:
            logger.info("OpenCL available — GPU-accelerated motion detection enabled")
        else:
            logger.warning("OpenCL not available — running on CPU")

        logger.info(
            f"MotionDetector ready | sensitivity={self.sensitivity} "
            f"threshold={self.motion_threshold}px | frame_skip={self.frame_skip} "
            f"| opencl={self.use_opencl}"
        )

    def detect_motion(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[Tuple[float, float]], dict]:
        logger.info(f"Starting motion detection: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        scale = self.detection_scale / orig_w
        det_w = self.detection_scale
        det_h = int(orig_h * scale)

        logger.info(
            f"Video: {total_frames} frames @ {fps:.2f} fps ({duration:.1f}s) "
            f"{orig_w}×{orig_h} → detection {det_w}×{det_h}"
        )

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Cannot read first frame")

        first_small = cv2.resize(first_frame, (det_w, det_h))
        avg_umat = cv2.UMat(first_small.astype(np.float32))

        motion_frames: List[int] = []
        frame_idx = 0
        processed = 0
        skipped_bad = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if progress_callback and frame_idx % 60 == 0:
                progress_callback((frame_idx / total_frames) * 100)

            if frame_idx % self.frame_skip != 0:
                continue

            if self._is_bad_frame(frame):
                skipped_bad += 1
                continue

            small = cv2.resize(frame, (det_w, det_h))
            umat = cv2.UMat(small.astype(np.float32))

            cv2.accumulateWeighted(umat, avg_umat, self.bg_learn_rate)
            diff = cv2.absdiff(umat, avg_umat)

            diff_uint8 = cv2.convertScaleAbs(diff)

            gray = cv2.cvtColor(diff_uint8, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)

            motion_pixels = cv2.countNonZero(dilated)

            if motion_pixels > self.motion_threshold:
                motion_frames.append(frame_idx)

            processed += 1

        cap.release()

        motion_segments = self._frames_to_segments(
            motion_frames, fps, self.min_motion_duration, self.segment_padding, total_frames
        )

        total_motion = sum(e - s for s, e in motion_segments)
        logger.info(
            f"Detection complete: {len(motion_segments)} segments, "
            f"{total_motion:.1f}s motion out of {duration:.1f}s "
            f"({total_motion / duration * 100:.1f}%) | "
            f"processed {processed} frames, skipped {skipped_bad} bad frames"
        )

        stats = {
            "total_frames": total_frames,
            "processed_frames": processed,
            "skipped_bad": skipped_bad,
            "motion_segments": len(motion_segments),
            "motion_duration": total_motion,
            "video_duration": duration,
            "detection_method": "OpenCL UMat" if self.use_opencl else "CPU",
            "input_file": video_path,
        }
        return motion_segments, stats

    @staticmethod
    def _is_bad_frame(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        return mean < 8 or mean > 248

    def _frames_to_segments(
        self,
        motion_frames: List[int],
        fps: float,
        min_duration: float,
        padding: float,
        total_frames: int,
    ) -> List[Tuple[float, float]]:
        if not motion_frames:
            return []

        merge_gap_frames = fps * 2.0
        padding_frames = int(padding * fps)

        raw_segments: List[Tuple[float, float]] = []
        seg_start = motion_frames[0]
        seg_end = motion_frames[0]

        for f in motion_frames[1:]:
            if f - seg_end <= merge_gap_frames:
                seg_end = f
            else:
                raw_segments.append((seg_start, seg_end))
                seg_start = seg_end = f
        raw_segments.append((seg_start, seg_end))

        padded: List[Tuple[float, float]] = []
        for s, e in raw_segments:
            t_start = max(0.0, (s - padding_frames) / fps)
            t_end = min(total_frames / fps, (e + padding_frames) / fps)
            if t_end - t_start >= min_duration:
                padded.append((t_start, t_end))

        if not padded:
            return []
        merged = [padded[0]]
        for s, e in padded[1:]:
            if s <= merged[-1][1] + 1.0:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        return merged