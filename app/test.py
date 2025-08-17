# tests.py
import pytest
from video_processing import compute_motion_score, is_white_or_black_frame

def test_compute_motion_score():
    frame1 = np.zeros((100,100,3), dtype=np.uint8)
    frame2 = np.ones((100,100,3), dtype=np.uint8) * 255
    score = compute_motion_score(frame1, frame2, 0)
    assert score > 0

def test_is_white_or_black_frame():
    black = np.zeros((100,100,3), dtype=np.uint8)
    assert is_white_or_black_frame(black, 240, 10)
    white = np.ones((100,100,3), dtype=np.uint8) * 255
    assert is_white_or_black_frame(white, 240, 10)
    gray = np.ones((100,100,3), dtype=np.uint8) * 128
    assert not is_white_or_black_frame(gray, 240, 10)