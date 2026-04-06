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
    def __init__(
        self,
        sensitivity: float = 0.5,
        min_area: int = 500,
        algorithm: str = "mog2",
        use_gpu: bool = False,
    ):
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.algorithm = algorithm
        self.use_gpu = use_gpu and self._cuda_available()
        self.prev_frame = None

        # Initialize the appropriate background subtractor with optimized settings
        if algorithm == "cnt":
            try:
                self.subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(
                    minPixelStability=15,
                    maxPixelStability=15 * 60,
                    useHistory=True,
                    isParallel=True,
                )
            except AttributeError:
                self.subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=30, detectShadows=False
                )
        elif algorithm == "knn":
            if self.use_gpu and hasattr(cv2.cuda, "createBackgroundSubtractorMOG2"):
                try:
                    self.subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                        history=500, varThreshold=30, detectShadows=False
                    )
                except Exception:
                    self.subtractor = cv2.createBackgroundSubtractorKNN(
                        history=500,
                        dist2Threshold=400.0,
                        detectShadows=False,
                    )
            else:
                self.subtractor = cv2.createBackgroundSubtractorKNN(
                    history=500,
                    dist2Threshold=400.0,
                    detectShadows=False,
                )
        elif algorithm == "simple":
            self.prev_frame = None
        else:  # mog2 (default) - optimized for bird detection
            if self.use_gpu and hasattr(cv2.cuda, "createBackgroundSubtractorMOG2"):
                try:
                    self.subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                        history=500,
                        varThreshold=30,
                        detectShadows=False,
                    )
                except Exception:
                    self.subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500,
                        varThreshold=30,
                        detectShadows=False,
                    )
            else:
                self.subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=30,
                    detectShadows=False,
                )

    def _cuda_available(self) -> bool:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

    def _preprocess_frame_cuda(
        self, frame: np.ndarray, downscale_factor: float = 0.25
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        new_width = int(width * downscale_factor)
        new_height = int(height * downscale_factor)
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        resized = cv2.cuda.resize(
            gpu_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cuda.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        result = gray.download()
        result = cv2.GaussianBlur(result, (7, 7), 0)
        return result

    def preprocess_frame(
        self, frame: np.ndarray, downscale_factor: float = 0.25
    ) -> np.ndarray:
        """Preprocess frame for optimal motion detection"""
        if self.use_gpu:
            try:
                return self._preprocess_frame_cuda(frame, downscale_factor)
            except Exception:
                pass

        height, width = frame.shape[:2]
        new_width = int(width * downscale_factor)
        new_height = int(height * downscale_factor)
        small = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        # Convert to grayscale
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        return gray

    def detect_motion(
        self, frame: np.ndarray, preprocess: bool = True
    ) -> Tuple[bool, List]:
        if self.algorithm == "simple":
            return self._detect_motion_simple(frame, preprocess)
        else:
            return self._detect_motion_advanced(frame, preprocess)

    def _detect_motion_advanced(
        self, frame: np.ndarray, preprocess: bool = True
    ) -> Tuple[bool, List]:
        """Motion detection using background subtraction (MOG2/KNN/CNT)"""
        # Preprocess frame
        if preprocess:
            processed = self.preprocess_frame(frame)
        else:
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply background subtractor
        fg_mask = self.subtractor.apply(processed)

        # Apply morphology to reduce noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Threshold the mask
        thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_areas = []
        has_motion = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                has_motion = True
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))

        return has_motion, motion_areas

    def _detect_motion_simple(
        self, frame: np.ndarray, preprocess: bool = True
    ) -> Tuple[bool, List]:
        """Simple frame differencing motion detection (fast alternative)"""
        if preprocess:
            gray = self.preprocess_frame(frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_areas = []
        has_motion = False

        if self.prev_frame is not None:
            # Calculate frame difference
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

            # Dilate to fill holes
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    has_motion = True
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_areas.append((x, y, w, h))

        # Keep the current frame for next-difference detection
        self.prev_frame = gray
        return has_motion, motion_areas
