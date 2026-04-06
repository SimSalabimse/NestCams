"""
Enhanced Motion Detection Module
Detects motion while filtering out bad frames
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Callable, Optional

from frame_analyzer import SmartFrameFilter

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self, config: dict):
        self.config = config
        self.sensitivity = config.get('sensitivity', 5)
        self.min_motion_duration = config.get('min_motion_duration', 0.5)
        self.motion_threshold = config.get('motion_threshold', 25)
        self.blur_size = config.get('blur_size', 21)
        
        # Adjusted threshold based on sensitivity
        self.adjusted_threshold = self.motion_threshold * (11 - self.sensitivity) / 5
        
        # Frame filtering
        self.frame_filter = SmartFrameFilter(config)
        
        logger.info(f"Motion detector initialized with sensitivity={self.sensitivity}, "
                   f"threshold={self.adjusted_threshold:.1f}")
    
    def detect_motion(self, video_path: str, 
                     progress_callback: Optional[Callable] = None) -> Tuple[List[Tuple[float, float]], dict]:
        """
        Detect motion segments in video with quality filtering
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback for progress updates
        
        Returns:
            Tuple of (motion_segments, statistics)
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
        
        # Check first frame quality
        keep_frame, frame1, reason = self.frame_filter.filter_frame(frame1)
        while not keep_frame and ret:
            ret, frame1 = cap.read()
            if ret:
                keep_frame, frame1, reason = self.frame_filter.filter_frame(frame1)
        
        if not ret:
            raise ValueError("No valid frames found in video")
        
        # Convert to grayscale and blur
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (self.blur_size, self.blur_size), 0)
        
        motion_frames = []  # List of frame numbers with motion
        valid_frames = {0: True}  # Track valid frames
        frame_count = 0
        valid_frame_count = 0
        
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
            
            # Check frame quality
            keep_frame, frame2_processed, reason = self.frame_filter.filter_frame(frame2)
            
            if not keep_frame:
                valid_frames[frame_count] = False
                continue
            
            valid_frames[frame_count] = True
            valid_frame_count += 1
            
            # Convert to grayscale and blur
            gray2 = cv2.cvtColor(frame2_processed, cv2.COLOR_BGR2GRAY)
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
            if motion_pixels > 1000:
                motion_frames.append(frame_count)
            
            # Move to next frame
            gray1 = gray2
        
        cap.release()
        
        # Convert motion frames to time segments
        motion_segments = self._frames_to_segments(motion_frames, fps, 
                                                   self.min_motion_duration)
        
        # Get filtering statistics
        filter_stats = self.frame_filter.get_statistics()
        
        logger.info(f"Detected {len(motion_segments)} motion segments")
        total_motion_time = sum(end - start for start, end in motion_segments)
        logger.info(f"Total motion time: {total_motion_time:.1f}s out of {duration:.1f}s "
                   f"({total_motion_time/duration*100:.1f}%)")
        logger.info(f"Frame filtering: {filter_stats['filtered']} bad frames removed "
                   f"({filter_stats.get('filter_rate', 0):.1f}%)")
        
        stats = {
            'total_frames': total_frames,
            'valid_frames': valid_frame_count,
            'motion_segments': len(motion_segments),
            'motion_duration': total_motion_time,
            'video_duration': duration,
            'filter_stats': filter_stats
        }
        
        return motion_segments, stats
    
    def _frames_to_segments(self, motion_frames: List[int], fps: float, 
                           min_duration: float) -> List[Tuple[float, float]]:
        """
        Convert list of motion frames to time segments
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
                if start - current_end <= 2.0:
                    current_end = end
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged_segments.append((current_start, current_end))
        
        return merged_segments
