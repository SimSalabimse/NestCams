"""
GPU-Optimized Motion Detection Module
Uses CUDA MOG2 for 5-20x faster detection on NVIDIA GPUs
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Callable, Optional
from frame_analyzer import SmartFrameFilter

logger = logging.getLogger(__name__)


class GPUMotionDetector:
    """GPU-accelerated motion detector using CUDA MOG2"""
    
    def __init__(self, config: dict):
        self.config = config
        self.sensitivity = config.get('sensitivity', 5)
        self.min_motion_duration = config.get('min_motion_duration', 0.8)  # Increased for bird visits
        self.motion_threshold = config.get('motion_threshold', 40)  # MOG2 variance threshold
        self.detection_scale = config.get('detection_scale', 320)  # Width for detection (320x240)
        self.frame_skip = config.get('frame_skip', 4)  # Process every Nth frame
        self.segment_padding = config.get('segment_padding', 1.0)  # Seconds before/after motion
        
        # Frame filtering
        self.frame_filter = SmartFrameFilter(config)
        
        # Check for CUDA support
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0 and config.get('use_gpu', True)
        
        if self.use_gpu:
            logger.info("🚀 Using GPU (CUDA) for motion detection")
            self.fgbg = cv2.cuda.createBackgroundSubtractorMOG2(
                detectShadows=False,
                varThreshold=self.motion_threshold
            )
        else:
            logger.info("Using CPU for motion detection")
            self.fgbg = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False,
                varThreshold=self.motion_threshold
            )
        
        logger.info(f"Motion detector initialized: GPU={self.use_gpu}, "
                   f"scale={self.detection_scale}p, skip={self.frame_skip}")
    
    def detect_motion(self, video_path: str, 
                     progress_callback: Optional[Callable] = None) -> Tuple[List[Tuple[float, float]], dict]:
        """
        Detect motion segments using GPU-accelerated MOG2
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback for progress updates
        
        Returns:
            Tuple of (motion_segments, statistics)
        """
        logger.info(f"Starting GPU motion detection on: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        # Calculate detection resolution (maintain aspect ratio)
        aspect_ratio = height / width
        detection_width = self.detection_scale
        detection_height = int(detection_width * aspect_ratio)
        
        logger.info(f"Video: {total_frames} frames at {fps} fps ({duration:.1f}s), "
                   f"{width}x{height} → {detection_width}x{detection_height} for detection")
        
        motion_frames = []
        frame_count = 0
        processed_count = 0
        
        # GPU upload if using CUDA
        if self.use_gpu:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_mask = cv2.cuda_GpuMat()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            if progress_callback and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
            
            # Frame skipping for speed
            if frame_count % self.frame_skip != 0:
                continue
            
            # Quick quality check (skip obviously bad frames)
            keep_frame, frame_processed, reason = self.frame_filter.filter_frame(frame)
            if not keep_frame:
                continue
            
            # Downscale for faster processing
            small_frame = cv2.resize(frame_processed, (detection_width, detection_height))
            
            # Apply MOG2
            if self.use_gpu:
                # GPU path
                gpu_frame.upload(small_frame)
                gpu_mask = self.fgbg.apply(gpu_frame)
                fg_mask = gpu_mask.download()
            else:
                # CPU path
                fg_mask = self.fgbg.apply(small_frame)
            
            # Count motion pixels
            motion_pixels = cv2.countNonZero(fg_mask)
            motion_ratio = motion_pixels / (detection_width * detection_height)
            
            # Adaptive threshold based on sensitivity
            # Lower sensitivity = more sensitive (detects smaller motion)
            threshold_ratio = 0.01 * (11 - self.sensitivity)  # 0.01 at sensitivity 10, 0.10 at sensitivity 1
            
            if motion_ratio > threshold_ratio:
                motion_frames.append(frame_count)
            
            processed_count += 1
        
        cap.release()
        
        # Convert motion frames to time segments with padding
        motion_segments = self._frames_to_segments_with_padding(
            motion_frames, fps, self.min_motion_duration, self.segment_padding, total_frames
        )
        
        # Get filtering statistics
        filter_stats = self.frame_filter.get_statistics()
        
        logger.info(f"Detected {len(motion_segments)} motion segments from {processed_count} processed frames")
        total_motion_time = sum(end - start for start, end in motion_segments)
        logger.info(f"Total motion time: {total_motion_time:.1f}s out of {duration:.1f}s "
                   f"({total_motion_time/duration*100:.1f}%)")
        logger.info(f"Frame filtering: {filter_stats['filtered']} bad frames removed")
        
        stats = {
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'valid_frames': filter_stats['valid'],
            'motion_segments': len(motion_segments),
            'motion_duration': total_motion_time,
            'video_duration': duration,
            'filter_stats': filter_stats,
            'detection_method': 'GPU MOG2' if self.use_gpu else 'CPU MOG2',
            'input_file': video_path
        }
        
        return motion_segments, stats
    
    def _frames_to_segments_with_padding(self, motion_frames: List[int], fps: float, 
                                         min_duration: float, padding: float,
                                         total_frames: int) -> List[Tuple[float, float]]:
        """
        Convert motion frames to time segments with padding and merging
        
        Args:
            motion_frames: List of frame numbers with motion
            fps: Video frame rate
            min_duration: Minimum segment duration
            padding: Seconds to add before/after motion
            total_frames: Total frames in video
        
        Returns:
            List of (start_time, end_time) tuples
        """
        if not motion_frames:
            return []
        
        # Account for frame skipping
        motion_frames = [f * self.frame_skip for f in motion_frames]
        
        segments = []
        start_frame = motion_frames[0]
        prev_frame = motion_frames[0]
        
        # Merge frames into segments
        merge_gap = fps * 2.0  # Merge if gap < 2 seconds
        
        for frame in motion_frames[1:]:
            if frame - prev_frame > merge_gap:
                # End previous segment
                end_frame = prev_frame
                
                # Add padding
                padding_frames = int(padding * fps)
                start_frame_padded = max(0, start_frame - padding_frames)
                end_frame_padded = min(total_frames - 1, end_frame + padding_frames)
                
                start_time = start_frame_padded / fps
                end_time = end_frame_padded / fps
                
                # Only add if duration meets minimum
                if end_time - start_time >= min_duration:
                    segments.append((start_time, end_time))
                
                # Start new segment
                start_frame = frame
            
            prev_frame = frame
        
        # Add final segment
        padding_frames = int(padding * fps)
        start_frame_padded = max(0, start_frame - padding_frames)
        end_frame_padded = min(total_frames - 1, prev_frame + padding_frames)
        
        start_time = start_frame_padded / fps
        end_time = end_frame_padded / fps
        
        if end_time - start_time >= min_duration:
            segments.append((start_time, end_time))
        
        # Merge overlapping/close segments
        merged = []
        if segments:
            current_start, current_end = segments[0]
            
            for start, end in segments[1:]:
                if start <= current_end + 1.5:  # Merge if within 1.5s
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged.append((current_start, current_end))
        
        return merged


# Factory function for backwards compatibility
def MotionDetector(config: dict):
    """Factory function to create appropriate motion detector"""
    return GPUMotionDetector(config)
