import cv2
import numpy as np
from typing import List, Tuple, Optional
import GPUtil


class HardwareManager:
    def __init__(self):
        self.gpu_available = self._check_gpu()
        self.cpu_count = cv2.getNumberOfCPUs()
        self.threads = max(1, self.cpu_count - 1)  # Reserve one for UI

    def _check_gpu(self) -> bool:
        try:
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except:
            return False

    def get_optimal_settings(self) -> dict:
        settings = {
            "threads": self.threads,
            "gpu": self.gpu_available,
        }
        if self.gpu_available:
            settings["opencv_backend"] = cv2.CAP_FFMPEG  # Or CUDA if available
        return settings


class MotionDetector:
    def __init__(self, sensitivity: float = 0.5, min_area: int = 500):
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=True
        )

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, List]:
        fg_mask = self.subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_areas = []
        has_motion = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                has_motion = True
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))

        return has_motion, motion_areas
