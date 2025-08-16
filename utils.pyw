import logging
import os
from datetime import datetime
import psutil
import requests
import time
import threading
from tkinter import Toplevel, Label

try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network stability checks will be limited.")

VERSION = "10.0.0_beta"
UPDATE_CHANNELS = ["Stable", "Beta"]

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'processor_log.txt'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
session_log_file = os.path.join(log_dir, f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
session_logger = logging.getLogger('session')
session_handler = logging.FileHandler(session_log_file)
session_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
session_logger.addHandler(session_handler)
session_logger.setLevel(logging.INFO)

thread_lock = threading.Lock()
cancel_events = {}  # dict for per-task cancellation
pause_event = threading.Event()  # for pause/resume
pause_event.set()  # Initially not paused

BATCH_SIZE = 4
WORKER_PROCESSES = 2

def log_session(message):
    """Log a message to the session log file."""
    session_logger.info(message)

def check_system_specs():
    """Adjust processing parameters based on system specifications."""
    global BATCH_SIZE, WORKER_PROCESSES
    cpu_cores = os.cpu_count() or 1
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    if total_ram_gb < 8:
        BATCH_SIZE = max(1, cpu_cores // 2)
        WORKER_PROCESSES = 1
    elif total_ram_gb < 16:
        BATCH_SIZE = max(2, cpu_cores // 2)
        WORKER_PROCESSES = min(2, cpu_cores)
    elif total_ram_gb < 32:
        BATCH_SIZE = max(4, cpu_cores)
        WORKER_PROCESSES = min(4, cpu_cores)
    else:
        BATCH_SIZE = max(8, cpu_cores * 2)
        WORKER_PROCESSES = min(8, cpu_cores)

    log_session(f"System specs: CPU cores={cpu_cores}, RAM={total_ram_gb:.2f} GB")
    log_session(f"Configured: Batch size={BATCH_SIZE}, Workers={WORKER_PROCESSES}")

def check_disk_space(required_mb=500):
    """Check if there's enough disk space for processing."""
    total, used, free = psutil.disk_usage(os.path.dirname(os.path.abspath(__file__)))
    free_mb = free / (1024 * 1024)
    if free_mb < required_mb:
        log_session(f"Insufficient disk space: {free_mb:.2f} MB free")
        return False
    return True

class ToolTip:
    """Simple tooltip class for CustomTkinter widgets."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y = self.widget.winfo_pointerxy()
        self.tooltip = Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x+20}+{y+20}")
        label = Label(self.tooltip, text=self.text, bg="yellow", relief="solid", borderwidth=1, padx=5, pady=3)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None