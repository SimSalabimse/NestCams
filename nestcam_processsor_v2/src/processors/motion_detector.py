"""
Advanced motion detection with improved algorithms
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add the parent directory to the path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    # Try relative imports first (when run as module)
    from ..utils.logger import get_logger
except ImportError:
    # Fall back to absolute imports (when run directly by Streamlit)
    from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MotionResult:
    """Result of motion detection analysis"""

    frame_indices: List[int]
    motion_scores: List[float]
    avg_motion: float
    peak_motion: float
    motion_variance: float
    total_frames: int
    processed_frames: int


class MotionDetector:
    """Advanced motion detection with multiple algorithms"""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)

        # Algorithm parameters
        self.frame_step = 1  # Process every Nth frame
        self.block_size = 16  # Block size for motion estimation
        self.min_motion_area = 100  # Minimum area for motion detection

    def detect_motion(self, video_path: str, progress_callback=None) -> MotionResult:
        """
        Detect motion in video using multiple algorithms

        Args:
            video_path: Path to video file
            progress_callback: Optional progress callback function

        Returns:
            MotionResult object with detection results
        """
        self.logger.info(f"Starting motion detection for {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.logger.info(f"Video stats: {total_frames} frames, {fps} FPS")

        # Initialize motion detection
        frame_indices = []
        motion_scores = []

        # Read first few frames to establish baseline
        baseline_frames = self._get_baseline_frames(cap, num_frames=10)

        # Process frames with parallel processing for better performance
        with ThreadPoolExecutor(
            max_workers=self.config.processing.worker_processes
        ) as executor:
            futures = []
            frame_batch = []

            for frame_idx in range(0, total_frames, self.frame_step):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_batch.append((frame_idx, frame))

                # Process in batches
                if len(frame_batch) >= self.config.processing.batch_size:
                    future = executor.submit(
                        self._process_frame_batch, frame_batch, baseline_frames
                    )
                    futures.append(future)
                    frame_batch = []

                    if progress_callback:
                        progress = (frame_idx / total_frames) * 100
                        progress_callback(progress, frame_idx, total_frames)

            # Process remaining frames
            if frame_batch:
                future = executor.submit(
                    self._process_frame_batch, frame_batch, baseline_frames
                )
                futures.append(future)

            # Collect results
            for future in futures:
                batch_indices, batch_scores = future.result()
                frame_indices.extend(batch_indices)
                motion_scores.extend(batch_scores)

        cap.release()

        # Calculate statistics
        if motion_scores:
            avg_motion = np.mean(motion_scores)
            peak_motion = np.max(motion_scores)
            motion_variance = np.var(motion_scores)
        else:
            avg_motion = peak_motion = motion_variance = 0.0

        result = MotionResult(
            frame_indices=frame_indices,
            motion_scores=motion_scores,
            avg_motion=avg_motion,
            peak_motion=peak_motion,
            motion_variance=motion_variance,
            total_frames=total_frames,
            processed_frames=len(motion_scores),
        )

        self.logger.info(
            f"Motion detection completed: {len(frame_indices)} frames with motion"
        )
        return result

    def detect_motion_streaming(
        self, video_path: str, progress_callback=None
    ) -> MotionResult:
        """
        Memory-efficient motion detection that processes frames one by one
        without storing them in memory.
        """
        # Initialize with running average baseline (not stored frames)
        baseline_avg = None
        baseline_count = 0

        # Process frames one by one
        for frame_idx in range(0, total_frames, self.frame_step):
            ret, frame = cap.read()
            if not ret:
                break

            # Update running average baseline
            frame_resized = cv2.resize(frame, (640, 360))
            if baseline_avg is None:
                baseline_avg = frame_resized.astype(np.float32)
            else:
                baseline_avg = (baseline_avg * baseline_count + frame_resized) / (
                    baseline_count + 1
                )
            baseline_count += 1

            # Calculate motion score using current frame vs running average
            score = self._calculate_streaming_motion_score(frame_resized, baseline_avg)

            if score > self.config.processing.motion_threshold:
                frame_indices.append(frame_idx)
            motion_scores.append(score)

        # Return results without keeping frames in memory
        return MotionResult(
            frame_indices=frame_indices,
            motion_scores=motion_scores,
            avg_motion=np.mean(motion_scores),  # Approximate average for streaming
            peak_motion=np.max(motion_scores),  # Approximate peak for streaming
            motion_variance=np.var(motion_scores),  # Approximate variance for streaming
            total_frames=total_frames,
            processed_frames=len(motion_scores),
        )

    def _get_baseline_frames(
        self, cap: cv2.VideoCapture, num_frames: int = 10
    ) -> List[np.ndarray]:
        """Get baseline frames for motion detection"""
        baseline_frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
            baseline_frames.append(frame)

        # Restore original position
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

        return baseline_frames

    def _process_frame_batch(
        self,
        frame_batch: List[Tuple[int, np.ndarray]],
        baseline_frames: List[np.ndarray],
    ) -> Tuple[List[int], List[float]]:
        """Process a batch of frames for motion detection"""
        frame_indices = []
        motion_scores = []

        for frame_idx, frame in frame_batch:
            # Resize frame for consistent processing
            frame_resized = cv2.resize(frame, (640, 360))

            # Calculate motion score using multiple methods
            score = self._calculate_motion_score(frame_resized, baseline_frames)

            if score > self.config.processing.motion_threshold:
                frame_indices.append(frame_idx)

            motion_scores.append(score)

        return frame_indices, motion_scores

    def _calculate_motion_score(
        self, frame: np.ndarray, baseline_frames: List[np.ndarray]
    ) -> float:
        """Calculate motion score using multiple algorithms"""
        scores = []

        # Method 1: Frame differencing
        if baseline_frames:
            baseline_avg = np.mean(baseline_frames, axis=0).astype(np.uint8)
            diff = cv2.absdiff(frame, baseline_avg)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            score1 = np.sum(thresh > 0)
            scores.append(score1)

        # Method 2: Optical flow estimation
        try:
            score2 = self._optical_flow_score(frame, baseline_frames)
            scores.append(score2)
        except Exception as e:
            self.logger.debug(f"Optical flow failed: {e}")

        # Method 3: Edge detection based motion
        edges = cv2.Canny(frame, 100, 200)
        score3 = np.sum(edges > 0)
        scores.append(score3 / 10)  # Scale down

        # Method 4: Color histogram comparison
        if baseline_frames:
            hist_score = self._histogram_motion_score(frame, baseline_frames)
            scores.append(hist_score)

        # Return weighted average of all methods
        if scores:
            weights = [0.4, 0.3, 0.2, 0.1]  # Weights for different methods
            final_score = sum(score * weight for score, weight in zip(scores, weights))
            return final_score

        return 0.0

    def _optical_flow_score(
        self, frame: np.ndarray, baseline_frames: List[np.ndarray]
    ) -> float:
        """Calculate motion score using optical flow"""
        if not baseline_frames:
            return 0.0

        prev_frame = cv2.cvtColor(baseline_frames[-1], cv2.COLOR_BGR2GRAY)
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return np.sum(magnitude > 1.0)

    def _histogram_motion_score(
        self, frame: np.ndarray, baseline_frames: List[np.ndarray]
    ) -> float:
        """Calculate motion score using color histogram comparison"""
        if not baseline_frames:
            return 0.0

        # Calculate histogram for current frame
        hist_current = cv2.calcHist(
            [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)

        # Calculate histogram for baseline
        baseline_avg = np.mean(baseline_frames, axis=0).astype(np.uint8)
        hist_baseline = cv2.calcHist(
            [baseline_avg], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist_baseline, hist_baseline, 0, 1, cv2.NORM_MINMAX)

        # Compare histograms
        similarity = cv2.compareHist(hist_current, hist_baseline, cv2.HISTCMP_CORREL)
        return (1.0 - similarity) * 1000  # Convert to motion score
