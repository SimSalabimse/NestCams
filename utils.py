import logging
import logging.handlers
import os
from datetime import datetime
import subprocess
import requests
import queue
import threading
try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network stability checks limited")

import tkinter as tk

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'processor_log.txt'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Session logging with QueueHandler
session_log_file = os.path.join(log_dir, f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
session_logger = logging.getLogger('session')
session_logger.setLevel(logging.INFO)
log_queue = queue.Queue(-1)  # Unbounded queue for simplicity
queue_handler = logging.handlers.QueueHandler(log_queue)
session_logger.addHandler(queue_handler)

# File handler for writing to session log file
file_handler = logging.FileHandler(session_log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

def log_listener(queue, handler):
    """Listen to log queue and write to file."""
    while True:
        try:
            record = queue.get()
            if record is None:  # None is a signal to stop
                break
            handler.handle(record)
        except Exception as e:
            print(f"Log listener error: {e}")

# Start logging thread
listener_thread = threading.Thread(target=log_listener, args=(log_queue, file_handler), daemon=True)
listener_thread.start()

def log_session(message):
    """Log a message to the session log via queue."""
    session_logger.info(message)

def stop_logging():
    """Stop the logging thread gracefully."""
    log_queue.put(None)
    listener_thread.join(timeout=1.0)

class ToolTip:
    """Tooltip class for GUI usability."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="yellow", relief="solid", borderwidth=1, padx=5, pady=3)
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def validate_video_file(file_path):
    """Validate video file integrity."""
    try:
        cmd = ['ffmpeg', '-i', file_path, '-f', 'null', '-']
        subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        log_session(f"Validated video: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        log_session(f"Validation failed for {file_path}: {e.stderr.decode()}")
        return False

def check_network_stability():
    """Check network stability for uploads."""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code != 200:
            log_session(f"Network ping failed: Status {response.status_code}")
            return False
        if speedtest is None:
            log_session("No speedtest module, using ping only")
            return True
        st = speedtest.Speedtest()
        st.get_best_server()
        upload_speed = st.upload() / 1_000_000
        if upload_speed < 1.0:
            log_session(f"Upload speed too low: {upload_speed:.2f} Mbps")
            return False
        log_session(f"Network stable: Upload speed {upload_speed:.2f} Mbps")
        return True
    except Exception as e:
        log_session(f"Network check failed: {str(e)}")
        return False