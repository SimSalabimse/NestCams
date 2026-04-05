"""
Motion Detection Module
Detects motion in video files and returns time segments with activity
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Callable, Optional

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self, config: dict):
        self.config = config
        self.sensitivity = config.get('sensitivity', 5)
        self.min_motion_duration = config.get('min_motion_duration', 0.5)
        self.motion_threshold = config.get('motion_threshold', 25)
        self.blur_size = config.get('blur_size', 21)
        
        # Adjust threshold based on sensitivity (1-10 scale)
        # Higher sensitivity = lower threshold
        self.adjusted_threshold = self.motion_threshold * (11 - self.sensitivity) / 5
        
        logger.info(f"Motion detector initialized with sensitivity={self.sensitivity}, "
                   f"threshold={self.adjusted_threshold:.1f}")
    
    def detect_motion(self, video_path: str, 
                     progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
        """
        Detect motion segments in video
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of (start_time, end_time) tuples in seconds where motion was detected
        """
        logger.info(f"Starting motion detection on: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video info: {total_frames} frames at {fps} fps ({duration:.1f} seconds)")
        
        # Initialize
        ret, frame1 = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        # Convert to grayscale and blur
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (self.blur_size, self.blur_size), 0)
        
        motion_frames = []  # List of frame numbers with motion
        frame_count = 0
        
        # Process frames
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            if progress_callback and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
            
            # Convert to grayscale and blur
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (self.blur_size, self.blur_size), 0)
            
            # Compute difference
            frame_diff = cv2.absdiff(gray1, gray2)
            
            # Threshold
            _, thresh = cv2.threshold(frame_diff, self.adjusted_threshold, 255, cv2.THRESH_BINARY)
            
            # Dilate to fill gaps
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Count non-zero pixels (motion pixels)
            motion_pixels = cv2.countNonZero(dilated)
            
            # If significant motion detected
            if motion_pixels > 1000:  # Minimum pixel threshold
                motion_frames.append(frame_count)
            
            # Move to next frame
            gray1 = gray2
        
        cap.release()
        
        # Convert motion frames to time segments
        motion_segments = self._frames_to_segments(motion_frames, fps, 
                                                   self.min_motion_duration)
        
        logger.info(f"Detected {len(motion_segments)} motion segments")
        total_motion_time = sum(end - start for start, end in motion_segments)
        logger.info(f"Total motion time: {total_motion_time:.1f}s out of {duration:.1f}s "
                   f"({total_motion_time/duration*100:.1f}%)")
        
        return motion_segments
    
    def _frames_to_segments(self, motion_frames: List[int], fps: float, 
                           min_duration: float) -> List[Tuple[float, float]]:
        """
        Convert list of motion frames to time segments
        
        Args:
            motion_frames: List of frame numbers with detected motion
            fps: Video frame rate
            min_duration: Minimum duration for a segment in seconds
        
        Returns:
            List of (start_time, end_time) tuples
        """
        if not motion_frames:
            return []
        
        segments = []
        start_frame = motion_frames[0]
        prev_frame = motion_frames[0]
        
        # Merge consecutive frames into segments
        for frame in motion_frames[1:]:
            # If there's a gap larger than 1 second, start new segment
            if frame - prev_frame > fps:
                # End previous segment
                end_frame = prev_frame
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                # Only add if duration meets minimum
                if end_time - start_time >= min_duration:
                    segments.append((start_time, end_time))
                
                # Start new segment
                start_frame = frame
            
            prev_frame = frame
        
        # Add final segment
        end_frame = prev_frame
        start_time = start_frame / fps
        end_time = end_frame / fps
        if end_time - start_time >= min_duration:
            segments.append((start_time, end_time))
        
        # Merge nearby segments (within 2 seconds)
        merged_segments = []
        if segments:
            current_start, current_end = segments[0]
            
            for start, end in segments[1:]:
                if start - current_end <= 2.0:  # Merge if gap is small
                    current_end = end
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged_segments.append((current_start, current_end))
        
        return merged_segments


class GPUMotionDetector(MotionDetector):
    """GPU-accelerated motion detector using CUDA (if available)"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Check for CUDA support
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_cuda = True
                logger.info("CUDA acceleration available and enabled")
            else:
                self.use_cuda = False
                logger.info("CUDA not available, falling back to CPU")
        except:
            self.use_cuda = False
            logger.info("CUDA not available, falling back to CPU")
    
    def detect_motion(self, video_path: str, 
                     progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
        """GPU-accelerated motion detection"""
        
        if not self.use_cuda:
            # Fall back to CPU implementation
            return super().detect_motion(video_path, progress_callback)
        
        # TODO: Implement CUDA-accelerated version
        # For now, use CPU version
        logger.info("GPU motion detection not yet implemented, using CPU")
        return super().detect_motion(video_path, progress_callback)
