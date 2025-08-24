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
from pathlib import Path

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
        Optimized two-pass motion detection with configurable detail levels
        """
        # Get detailed analysis settings from config
        use_detailed = self.config.processing.use_detailed_analysis
        detail_level = self.config.processing.detail_level
        context_window = self.config.processing.context_window_size

        # Adjust settings based on detail level
        if detail_level == "light":
            second_pass_window = max(1, context_window // 2)  # Smaller context
            analysis_methods = ["motion_diff"]  # Only basic method
        elif detail_level == "normal":
            second_pass_window = context_window  # Standard context
            analysis_methods = ["white_threshold", "motion_diff"]  # Balanced methods
        else:  # detailed
            second_pass_window = context_window * 2  # Larger context
            analysis_methods = [
                "white_threshold",
                "black_threshold",
                "motion_diff",
                "edge_detection",
                "histogram",
            ]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        file_size_gb = Path(video_path).stat().st_size / (1024**3)

        # Adaptive frame sampling based on file size
        if file_size_gb > 40:  # 40GB+ files
            first_pass_step = 5  # Process every 5th frame (80% reduction)
            second_pass_window = 3  # Reduced from 10 to 3 frames context
        else:  # 20-40GB files
            first_pass_step = 4  # Process every 4th frame (75% reduction)
            second_pass_window = 2  # Reduced from 8 to 2 frames context

        if progress_callback:
            progress_callback(
                5,
                f"🚀 Starting optimized motion detection for {file_size_gb:.1f}GB file",
                {
                    "stage": "Initializing",
                    "file_size_gb": file_size_gb,
                    "first_pass_step": first_pass_step,
                    "second_pass_window": second_pass_window,
                },
            )

        # ===== PASS 1: Fast Motion Detection =====
        if progress_callback:
            progress_callback(
                15,
                f"🎯 Pass 1: Fast motion detection (every {first_pass_step}th frame)",
                {"stage": "Pass 1 - Fast Detection"},
            )

        # Initialize baseline for fast detection
        baseline_avg = None
        baseline_count = 0
        potential_motion_frames = []

        # Fast first pass - process every Nth frame
        for frame_idx in range(0, total_frames, first_pass_step):
            ret, frame = cap.read()
            if not ret:
                break

            # Quick motion detection (simplified algorithm)
            frame_resized = cv2.resize(frame, (320, 180))  # Smaller for speed

            # Update running baseline
            if baseline_avg is None:
                baseline_avg = frame_resized.astype(np.float32)
            else:
                baseline_avg = (baseline_avg * baseline_count + frame_resized) / (
                    baseline_count + 1
                )
            baseline_count += 1

            # Fast motion score calculation
            if baseline_count > 5:  # Wait for baseline to stabilize
                diff = cv2.absdiff(frame_resized, baseline_avg.astype(np.uint8))
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                motion_score = np.sum(gray_diff > 30)  # Simple threshold

                if motion_score > 500:  # Fast motion threshold
                    potential_motion_frames.append(frame_idx)

            if progress_callback and frame_idx % (first_pass_step * 10) == 0:
                progress = (frame_idx / total_frames) * 30  # 30% of total progress
                progress_callback(
                    15 + progress,
                    f"⚡ Fast scan: {len(potential_motion_frames)} potential motion areas | Frame: {frame_idx}/{total_frames}",
                    {
                        "stage": "Pass 1 - Fast Detection",
                        "current_frame": frame_idx,
                        "total_frames": total_frames,
                        "potential_motion": len(potential_motion_frames),
                        "progress": f"{progress:.1f}%",
                        "frames_per_second": (
                            f"{(frame_idx / max(1, time.time() - start_time)):.1f}"
                            if "start_time" in locals()
                            else "N/A"
                        ),
                    },
                )

        # ===== PASS 2: Detailed Analysis =====
        if progress_callback:
            progress_callback(
                45,
                f"🔍 Pass 2: Detailed analysis of {len(potential_motion_frames)} motion areas",
                {"stage": "Pass 2 - Detailed Analysis"},
            )

        # Expand motion frames with context windows
        detailed_frames = set()
        for motion_frame in potential_motion_frames:
            start_frame = max(0, motion_frame - second_pass_window)
            end_frame = min(total_frames, motion_frame + second_pass_window + 1)
            detailed_frames.update(range(start_frame, end_frame))

        # Sort and remove duplicates (set already handles this)
        detailed_frames = sorted(list(detailed_frames))

        # Detailed second pass with white/black thresholds
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        final_motion_frames = []
        final_motion_scores = []
        processed_count = 0

        for frame_idx in detailed_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Full detailed motion analysis with white/black thresholds
            frame_resized = cv2.resize(frame, (640, 360))

            # Calculate detailed motion score with multiple algorithms
            score = self._calculate_detailed_motion_score(frame_resized)

            if score > self.config.processing.motion_threshold:
                final_motion_frames.append(frame_idx)
                final_motion_scores.append(score)

            processed_count += 1

            if progress_callback and processed_count % 50 == 0:  # Changed from 20 to 50
                progress = (
                    45 + (processed_count / len(detailed_frames)) * 50
                )  # 50% more
                progress_callback(
                    progress,
                    f"🎨 Detailed analysis: {len(final_motion_frames)} motion events ({processed_count}/{len(detailed_frames)} frames)",
                    {
                        "stage": "Pass 2 - Detailed Analysis",
                        "processed": processed_count,
                        "total_detailed": len(detailed_frames),
                        "motion_events": len(final_motion_frames),
                        "white_threshold": self.config.processing.white_threshold,
                        "black_threshold": self.config.processing.black_threshold,
                    },
                )

        cap.release()

        # Calculate final statistics
        if final_motion_scores:
            avg_motion = np.mean(final_motion_scores)
            peak_motion = np.max(final_motion_scores)
            motion_variance = np.var(final_motion_scores)
        else:
            avg_motion = peak_motion = motion_variance = 0.0

        result = MotionResult(
            frame_indices=final_motion_frames,
            motion_scores=final_motion_scores,
            avg_motion=avg_motion,
            peak_motion=peak_motion,
            motion_variance=motion_variance,
            total_frames=total_frames,
            processed_frames=len(final_motion_scores),
        )

        if progress_callback:
            progress_callback(
                95,
                f"⚡ Using fast scan results: {len(potential_motion_frames)} motion areas detected",
                {
                    "stage": "Complete",
                    "motion_events": len(potential_motion_frames),
                    "processing_reduction": f"{(1 - len(potential_motion_frames)/total_frames)*100:.1f}%",
                },
            )

        # Skip detailed analysis if disabled
        if not use_detailed:
            if progress_callback:
                progress_callback(
                    95,
                    f"⚡ Fast scan complete: {len(potential_motion_frames)} motion areas detected",
                    {
                        "stage": "Complete",
                        "motion_events": len(potential_motion_frames),
                        "processing_reduction": f"{(1 - len(potential_motion_frames)/total_frames)*100:.1f}%",
                    },
                )

            # Return fast scan results
            result = MotionResult(
                frame_indices=potential_motion_frames,
                motion_scores=[500.0] * len(potential_motion_frames),
                avg_motion=500.0,
                peak_motion=500.0,
                motion_variance=0.0,
                total_frames=total_frames,
                processed_frames=len(potential_motion_frames),
            )
            return result

        # Return results from detailed analysis
        result = MotionResult(
            frame_indices=final_motion_frames,
            motion_scores=final_motion_scores,
            avg_motion=avg_motion,
            peak_motion=peak_motion,
            motion_variance=motion_variance,
            total_frames=total_frames,
            processed_frames=len(final_motion_scores),
        )

        self.logger.info(
            f"Optimized motion detection completed: {len(final_motion_frames)} frames with motion "
            f"(processed {len(detailed_frames)}/{total_frames} frames, "
            f"{(1 - len(detailed_frames)/total_frames)*100:.1f}% reduction)"
        )
        return result

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

    def _calculate_streaming_motion_score(
        self, frame: np.ndarray, baseline_avg: np.ndarray
    ) -> float:
        """Calculate motion score using streaming baseline"""
        # Uses running average baseline instead of stored frames

        # Calculate histogram for current frame
        hist_current = cv2.calcHist(
            [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)

        # Convert baseline_avg to uint8 for histogram calculation
        baseline_avg_uint8 = baseline_avg.astype(np.uint8)

        # Calculate histogram for baseline
        hist_baseline = cv2.calcHist(
            [baseline_avg_uint8], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist_baseline, hist_baseline, 0, 1, cv2.NORM_MINMAX)

        # Compare histograms
        similarity = cv2.compareHist(hist_current, hist_baseline, cv2.HISTCMP_CORREL)
        return (1.0 - similarity) * 1000  # Convert to motion score

    def _calculate_detailed_motion_score(self, frame: np.ndarray) -> float:
        """Configurable detailed motion score with selectable analysis methods"""
        scores = []
        methods = self.config.processing.analysis_methods

        if "white_threshold" in methods:
            # 1. White threshold analysis (bright areas)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, white_mask = cv2.threshold(
                gray, self.config.processing.white_threshold, 255, cv2.THRESH_BINARY
            )
            white_score = np.sum(white_mask > 0)
            scores.append(white_score * 0.3)

        if "black_threshold" in methods:
            # 2. Black threshold analysis (dark areas)
            _, black_mask = cv2.threshold(
                gray, self.config.processing.black_threshold, 255, cv2.THRESH_BINARY_INV
            )
            black_score = np.sum(black_mask > 0)
            scores.append(black_score * 0.2)

        if "edge_detection" in methods:
            # 3. Edge detection (motion boundaries)
            edges = cv2.Canny(frame, 100, 200)
            edge_score = np.sum(edges > 0)
            scores.append(edge_score * 0.2)

        if "histogram" in methods:
            # 4. Color histogram comparison (scene changes)
            hist = cv2.calcHist(
                [frame], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256]
            )
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            hist_score = np.sum(hist > 0.01)
            scores.append(hist_score * 0.3)

        if "motion_diff" in methods:
            # 5. Simple motion difference (always included for reliability)
            if hasattr(self, "_previous_frame") and self._previous_frame is not None:
                diff = cv2.absdiff(frame, self._previous_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                motion_score = np.sum(gray_diff > 30)
                scores.append(motion_score * 0.4)

        # Store current frame for next comparison
        self._previous_frame = frame.copy()

        return sum(scores) if scores else 0.0
