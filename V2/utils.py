"""
utils.py — Logging, tooltips, validation helpers
═══════════════════════════════════════════════════════════════════════════════

LOGGING DESIGN
──────────────
Two complementary log streams:

1. Root logger (processor_log.txt, DEBUG level)
   Every debug/info/warning/error from all modules via standard Python logging.
   Rotating: max 5 MB, 3 backups — so a 24 h video session never fills disk.

2. Session logger (session_YYYYMMDD_HHMMSS.txt, INFO level)
   Human-readable per-session summary written asynchronously via QueueHandler.
   Easier to share for bug reports.

Both loggers are configured once at import time.  Modules use the standard
    logger = logging.getLogger(__name__)
pattern and their output flows into processor_log.txt automatically.

For per-stage structured entries, use:
    from utils import log_session
    log_session("Stage 2 complete | segments=7 motion=45.2s")
"""

import logging
import logging.handlers
import os
import queue
import subprocess
import threading
from datetime import datetime

import requests
import tkinter as tk

try:
    import speedtest as _speedtest
except ImportError:
    _speedtest = None

# ── Directory ──────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(_LOG_DIR, exist_ok=True)

# ── 1. Root rotating file handler (processor_log.txt) ─────────────────────────
_root_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_LOG_DIR, "processor_log.txt"),
    maxBytes=5 * 1024 * 1024,   # 5 MB
    backupCount=3,
    encoding="utf-8",
)
_root_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_root_handler.setLevel(logging.DEBUG)

# Also echo WARNING+ to stderr so developers see errors in the terminal
_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.WARNING)
_stderr_handler.setFormatter(logging.Formatter("%(levelname)s  %(name)s: %(message)s"))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[_root_handler, _stderr_handler],
)

# ── 2. Session logger (per-run file, async) ────────────────────────────────────
_session_file = os.path.join(
    _LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
_session_logger = logging.getLogger("session")
_session_logger.setLevel(logging.INFO)
_session_logger.propagate = False   # don't send to root handler

_log_q: queue.Queue = queue.Queue(-1)
_q_handler = logging.handlers.QueueHandler(_log_q)
_session_logger.addHandler(_q_handler)

_file_handler = logging.FileHandler(_session_file, encoding="utf-8")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
)


def _listener(q: queue.Queue, handler: logging.Handler) -> None:
    while True:
        record = q.get()
        if record is None:
            break
        handler.handle(record)


_listener_thread = threading.Thread(
    target=_listener, args=(_log_q, _file_handler), daemon=True
)
_listener_thread.start()


def log_session(message: str) -> None:
    """Write a line to the current session log."""
    _session_logger.info(message)


def stop_logging() -> None:
    """Flush and close the session log. Call on app exit."""
    _log_q.put(None)
    _listener_thread.join(timeout=2.0)


# ── Tooltip ────────────────────────────────────────────────────────────────────

class ToolTip:
    """Hover tooltip for any Tkinter/CustomTkinter widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text   = text
        self._tip: "tk.Toplevel | None" = None
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
            padx=6, pady=3, font=("Segoe UI", 9), justify="left",
            wraplength=300,
        ).pack()

    def _hide(self, _=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ── Video validation ───────────────────────────────────────────────────────────

def validate_video_file(path: str) -> bool:
    """Return True if FFmpeg can read the file without error."""
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True,
        )
        log_session(f"Validated: {path}")
        return True
    except subprocess.CalledProcessError as exc:
        log_session(
            f"Validation FAILED: {path}\n"
            + exc.stderr.decode(errors="replace")[:300]
        )
        return False


# ── Network check ──────────────────────────────────────────────────────────────

def check_network_stability() -> bool:
    """Return True if internet is reachable."""
    try:
        resp = requests.get("https://www.google.com", timeout=5)
        if resp.status_code != 200:
            log_session(f"Network ping failed: HTTP {resp.status_code}")
            return False
        if _speedtest is None:
            log_session("Network OK (ping only; speedtest-cli not installed)")
            return True
        st  = _speedtest.Speedtest()
        st.get_best_server()
        mbps = st.upload() / 1_000_000
        if mbps < 1.0:
            log_session(f"Upload too slow: {mbps:.2f} Mbps")
            return False
        log_session(f"Network stable: {mbps:.2f} Mbps upload")
        return True
    except Exception as exc:
        log_session(f"Network check failed: {exc}")
        return False
