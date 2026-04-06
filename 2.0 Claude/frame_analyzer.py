"""
Frame Analyzer Module
Filters out bad frames: white/black, corrupted, blurry, non-nest content
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class FrameAnalyzer:
    """Analyzes and filters video frames for quality"""
    
    def __init__(self, config: dict):
        self.config = config
        self.white_threshold = config.get('white_threshold', 240)
        self.black_threshold = config.get('black_threshold', 15)
        self.blur_threshold = config.get('blur_threshold', 100.0)
        self.corruption_threshold = config.get('corruption_threshold', 0.3)
        
        logger.info("Frame analyzer initialized")
    
    def is_frame_valid(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Check if frame is valid for time-lapse
        
        Args:
            frame: OpenCV frame (BGR)
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for white frames (overexposed/pause screen)
        if self._is_white_frame(frame):
            return False, "white_frame"
        
        # Check for black frames (underexposed/stream down)
        if self._is_black_frame(frame):
            return False, "black_frame"
        
        # Check for corrupted frames (green/purple artifacts)
        if self._is_corrupted(frame):
            return False, "corrupted"
        
        # Check for blur (out of focus)
        if self._is_too_blurry(frame):
            return False, "blurry"
        
        return True, "valid"
    
    def _is_white_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is mostly white"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        white_pixel_ratio = np.sum(gray > self.white_threshold) / gray.size
        
        # If >90% of pixels are white or mean brightness >250
        return white_pixel_ratio > 0.9 or mean_brightness > 250
    
    def _is_black_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is mostly black"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        black_pixel_ratio = np.sum(gray < self.black_threshold) / gray.size
        
        # If >90% of pixels are black or mean brightness <10
        return black_pixel_ratio > 0.9 or mean_brightness < 10
    
    def _is_corrupted(self, frame: np.ndarray) -> bool:
        """
        Detect corrupted frames (green/purple screen, obs crash, stream corruption)
        """
        # Check for abnormal color distribution
        b, g, r = cv2.split(frame)
        
        # Corrupted frames often have one channel dominate
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)
        
        total_mean = (mean_b + mean_g + mean_r) / 3
        
        # If one channel is >2x the average, likely corrupted
        if mean_g > total_mean * 2.5:  # Green corruption
            return True
        if mean_b > total_mean * 2.5 and mean_r > total_mean * 2.5:  # Purple
            return True
        
        # Check for split screen corruption (half green/purple)
        h, w = frame.shape[:2]
        top_half = frame[:h//2, :]
        bottom_half = frame[h//2:, :]
        
        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)
        
        # If halves are drastically different
        if abs(top_mean - bottom_mean) > 100:
            return True
        
        # Check for abnormal variance (solid color screens)
        variance = np.var(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if variance < 50:  # Very low variance = likely solid color
            # But not if it's a valid dark/bright frame
            mean_val = np.mean(frame)
            if 30 < mean_val < 220:  # Mid-range but no variance = corruption
                return True
        
        return False
    
    def _is_too_blurry(self, frame: np.ndarray) -> bool:
        """
        Check if frame is too blurry using Laplacian variance
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Lower variance = more blur
        return laplacian_var < self.blur_threshold
    
    def calculate_exposure(self, frame: np.ndarray) -> float:
        """
        Calculate frame exposure/brightness
        
        Returns:
            Average brightness (0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def normalize_exposure(self, frame: np.ndarray, target_exposure: float = 128.0) -> np.ndarray:
        """
        Normalize frame exposure to reduce flickering
        
        Args:
            frame: Input frame
            target_exposure: Target average brightness
        
        Returns:
            Exposure-normalized frame
        """
        current_exposure = self.calculate_exposure(frame)
        
        if current_exposure < 10:  # Too dark, skip
            return frame
        
        # Calculate adjustment factor
        adjustment = target_exposure / current_exposure
        
        # Limit adjustment to prevent over-correction
        adjustment = np.clip(adjustment, 0.5, 2.0)
        
        # Apply adjustment
        normalized = cv2.convertScaleAbs(frame, alpha=adjustment, beta=0)
        
        return normalized
    
    def apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply color correction for more natural colors
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight saturation boost
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)  # 20% saturation boost
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return corrected
    
    def get_frame_statistics(self) -> dict:
        """Get statistics about filtered frames"""
        return {
            'white_frames': 0,
            'black_frames': 0,
            'corrupted_frames': 0,
            'blurry_frames': 0,
            'total_filtered': 0
        }


class SmartFrameFilter:
    """Intelligent frame filtering with learning"""
    
    def __init__(self, config: dict):
        self.config = config
        self.analyzer = FrameAnalyzer(config)
        self.frame_stats = {
            'total': 0,
            'filtered': 0,
            'white': 0,
            'black': 0,
            'corrupted': 0,
            'blurry': 0,
            'valid': 0
        }
        
        # Store recent exposure values for smoothing
        self.exposure_history = []
        self.exposure_window = 30  # frames
    
    def filter_frame(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, str]:
        """
        Filter and process frame
        
        Args:
            frame: Input frame
        
        Returns:
            Tuple of (keep_frame, processed_frame, reason)
        """
        self.frame_stats['total'] += 1
        
        # Check validity
        is_valid, reason = self.analyzer.is_frame_valid(frame)
        
        if not is_valid:
            self.frame_stats['filtered'] += 1
            self.frame_stats[reason.replace('_frame', '').replace('_', '')] += 1
            return False, frame, reason
        
        # Frame is valid, process it
        processed = frame.copy()
        
        # Calculate target exposure from recent history
        current_exposure = self.analyzer.calculate_exposure(frame)
        self.exposure_history.append(current_exposure)
        
        if len(self.exposure_history) > self.exposure_window:
            self.exposure_history.pop(0)
        
        # Use moving average as target
        if len(self.exposure_history) > 5:
            target_exposure = np.median(self.exposure_history)
            processed = self.analyzer.normalize_exposure(processed, target_exposure)
        
        # Apply color correction
        if self.config.get('color_correction', True):
            processed = self.analyzer.apply_color_correction(processed)
        
        self.frame_stats['valid'] += 1
        return True, processed, 'valid'
    
    def get_statistics(self) -> dict:
        """Get filtering statistics"""
        stats = self.frame_stats.copy()
        if stats['total'] > 0:
            stats['filter_rate'] = (stats['filtered'] / stats['total']) * 100
            stats['keep_rate'] = (stats['valid'] / stats['total']) * 100
        return stats
