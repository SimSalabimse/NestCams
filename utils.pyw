import logging
from datetime import datetime
import psutil
import requests
import time
import tkinter as tk
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

thread_lock = tk.Lock()
cancel_events = {}  # dict for per-task cancellation
pause_event = tk.Event()  # for pause/resume
pause_event.set()  # Initially not paused

BATCH_SIZE = 4
WORKER_PROCESSES = 2

def log_session(message):
    """Log a message to the session log file."""
    session_logger.info(message)

def check_system_specs():
    """Adjust processing parameters based on system specifications."""
    global BATCH_SIZE, WORKER_PROCESSES
    # Same as original

def check_disk_space(required_mb=500):
    """Check if there's enough disk space for processing."""
    # Same as original

class ToolTip:
    """Simple tooltip class for CustomTkinter widgets."""
    # Same as original