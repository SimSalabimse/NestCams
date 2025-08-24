"""
Video enhancement algorithms for improving video quality
"""

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

import cv2
import numpy as np


class VideoEnhancer:
    """Video enhancement processor with multiple algorithms"""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)

        # Enhancement settings from config
        self.clip_limit = config.processing.clip_limit
        self.saturation_multiplier = config.processing.saturation_multiplier
        self.white_threshold = config.processing.white_threshold
        self.black_threshold = config.processing.black_threshold

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a single video frame

        Args:
            frame: Input frame as numpy array

        Returns:
            Enhanced frame
        """
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels back
            enhanced_lab = cv2.merge([l, a, b])

            # Convert back to BGR
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Adjust saturation
            hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_multiplier, 0, 255)
            enhanced_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Apply brightness/contrast adjustment
            enhanced_frame = self._adjust_brightness_contrast(enhanced_frame)

            return enhanced_frame

        except Exception as e:
            self.logger.error(f"Error enhancing frame: {e}")
            return frame  # Return original frame if enhancement fails

    def _adjust_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast of the frame"""
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)

        # Calculate current brightness
        brightness = np.mean(frame_float)

        # Target brightness (slight increase)
        target_brightness = min(brightness * 1.1, 200)

        # Adjust brightness
        if brightness > 0:
            factor = target_brightness / brightness
            frame_float = np.clip(frame_float * factor, 0, 255)

        return frame_float.astype(np.uint8)

    def enhance_video(self, input_path: str, output_path: str) -> bool:
        """
        Enhance an entire video file

        Args:
            input_path: Path to input video
            output_path: Path to save enhanced video

        Returns:
            True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Enhance frame
                enhanced_frame = self.enhance_frame(frame)

                # Write enhanced frame
                writer.write(enhanced_frame)
                frame_count += 1

            cap.release()
            writer.release()

            self.logger.info(f"Enhanced video saved: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error enhancing video: {e}")
            return False
