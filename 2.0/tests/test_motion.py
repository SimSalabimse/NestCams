import cv2
import numpy as np
from core.motion_detector import MotionDetector


def test_motion_detection():
    detector = MotionDetector()

    # Create a test frame with some motion
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

    has_motion, areas = detector.detect_motion(frame)
    assert has_motion == True
    assert len(areas) > 0
    print("Motion detection test passed")


if __name__ == "__main__":
    test_motion_detection()
