"""
utils.py — Logging, tooltips, validation, and utility helpers  v2.0
"""

import atexit
import glob
import logging
import logging.handlers
import os
import queue
import random
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
import tkinter as tk

try:
    import speedtest as _speedtest
except ImportError:
    _speedtest = None

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(_LOG_DIR, exist_ok=True)

_root_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_LOG_DIR, "processor_log.txt"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_root_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_root_handler.setLevel(logging.DEBUG)

_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.WARNING)
_stderr_handler.setFormatter(logging.Formatter("%(levelname)s  %(name)s: %(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[_root_handler, _stderr_handler])

_session_file = os.path.join(
    _LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
_session_logger = logging.getLogger("session")
_session_logger.setLevel(logging.INFO)
_session_logger.propagate = False

_log_q: queue.Queue = queue.Queue(-1)
_q_handler = logging.handlers.QueueHandler(_log_q)
_session_logger.addHandler(_q_handler)

_file_handler = logging.FileHandler(_session_file, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))


def _listener(q: queue.Queue, handler: logging.Handler) -> None:
    while True:
        record = q.get()
        if record is None: break
        handler.handle(record)


_listener_thread = threading.Thread(target=_listener, args=(_log_q, _file_handler), daemon=True)
_listener_thread.start()


def log_session(message: str) -> None:
    _session_logger.info(message)


def stop_logging() -> None:
    _log_q.put(None)
    _listener_thread.join(timeout=2.0)


atexit.register(stop_logging)

logger = logging.getLogger(__name__)


class ToolTip:
    """Hover tooltip for any Tkinter/CustomTkinter widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text   = text
        self._tip   = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _=None) -> None:
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=self.text,
            background="#ffffe0", relief="solid", borderwidth=1,
            padx=6, pady=3, font=("Segoe UI", 9), justify="left", wraplength=300,
        ).pack()

    def _hide(self, _=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


def get_video_info(path: str) -> dict:
    try:
        import json as _json
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,codec_name:format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=15,
        )
        data   = _json.loads(r.stdout)
        stream = data.get("streams", [{}])[0]
        fmt    = data.get("format", {})
        fps_n, fps_d = stream.get("r_frame_rate", "30/1").split("/")
        return {
            "fps":      float(fps_n) / max(float(fps_d), 1),
            "duration": float(fmt.get("duration", 0)),
            "width":    int(stream.get("width", 0)),
            "height":   int(stream.get("height", 0)),
            "codec":    stream.get("codec_name", "unknown"),
        }
    except Exception as exc:
        logger.warning(f"get_video_info failed: {exc}")
        return {}


def validate_video_file(path: str) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True,
        )
        log_session(f"Validated: {path}")
        return True
    except subprocess.CalledProcessError as exc:
        log_session(f"Validation FAILED: {path}\n" + exc.stderr.decode(errors="replace")[:300])
        return False


def is_video_corrupt(path: str) -> bool:
    import cv2 as _cv2
    try:
        cap   = _cv2.VideoCapture(path)
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release(); return True
        import random as _r
        positions = [_r.randint(0, total - 1) for _ in range(10)]
        failed = 0
        for pos in positions:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, pos)
            ok, _ = cap.read()
            if not ok: failed += 1
        cap.release()
        return failed > 5
    except Exception:
        return True


def calculate_optimal_frame_skip(duration_sec: float, target_motion_frames: int = 5000) -> int:
    total_frames = duration_sec * 30
    skip = max(1, int(total_frames / max(target_motion_frames, 1)))
    return min(skip, 8)


def adaptive_sensitivity(average_brightness: float) -> int:
    if average_brightness < 50:   return 3
    if average_brightness < 100:  return 4
    return 5


def generate_unique_output_name(base: str, suffix: str) -> str:
    candidate = f"{base}_{suffix}.mp4"
    n = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix}({n}).mp4"; n += 1
    return candidate


def crash_recovery_marker(output_path: str) -> None:
    marker = output_path + ".processing.tmp"
    try:
        Path(marker).write_text(
            f"started={datetime.now().isoformat()}\noutput={output_path}\n"
        )
    except OSError:
        pass


def clear_recovery_marker(output_path: str) -> None:
    try: os.remove(output_path + ".processing.tmp")
    except OSError: pass


def resume_from_marker(folder: str) -> List[str]:
    markers = glob.glob(os.path.join(folder, "*.processing.tmp"))
    return [m.replace(".processing.tmp", "") for m in markers]


def check_network_stability() -> bool:
    try:
        resp = requests.get("https://www.google.com", timeout=5)
        if resp.status_code != 200:
            log_session(f"Network ping failed: HTTP {resp.status_code}")
            return False
        if _speedtest is None:
            log_session("Network OK (ping only)")
            return True
        st   = _speedtest.Speedtest()
        st.get_best_server()
        mbps = st.upload() / 1_000_000
        if mbps < 1.0:
            log_session(f"Upload too slow: {mbps:.2f} Mbps"); return False
        log_session(f"Network stable: {mbps:.2f} Mbps")
        return True
    except Exception as exc:
        log_session(f"Network check failed: {exc}")
        return False


BIRD_FACTS = [
    "Blue tits can remember thousands of food-hiding spots 🐦",
    "A robin's territory is typically just one hectare 🌿",
    "Great tits adjust their song pitch to cut through city noise 🎵",
    "Swallows can sleep mid-flight during migration ✈️",
    "House sparrows have been found nesting inside airports 🏗️",
    "Blackbirds can sing over 100 distinct song phrases 🎶",
    "A clutch of blue tit eggs can weigh more than the mother 💪",
    "Nest box occupancy rises by 40% when boxes face east or north 🧭",
    "Starlings mimic car alarms, phones, and other birds 🎭",
    "Baby birds grow 10× their hatch weight in the first week 🥚",
]


def random_bird_fact() -> str:
    return random.choice(BIRD_FACTS)
