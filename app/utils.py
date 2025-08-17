# utils.py
import logging
from logging.handlers import QueueHandler
import queue
import threading
import os
import subprocess
import json
import psutil
import requests
from packaging import version
import schedule
import time
import atexit
import tempfile
import shutil
try:
    import speedtest
except ImportError:
    speedtest = None

# Logging setup
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return log_dir

log_dir = setup_logging()
session_queue = queue.Queue()
session_logger = logging.getLogger('session')
session_handler = QueueHandler(session_queue)
session_logger.addHandler(session_handler)
session_logger.setLevel(logging.DEBUG)

def log_session(message):
    session_queue.put(message)

def process_session_logs():
    while not session_queue.empty():
        msg = session_queue.get()
        session_logger.debug(msg)

log_thread = threading.Thread(target=process_session_logs, daemon=True)
log_thread.start()

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = ctk.CTkToplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = ctk.CTkLabel(self.tooltip, text=self.text, justify='left')
        label.pack(ipadx=2, ipady=2)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def validate_video_file(file_path):
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-i", file_path, "-f", "null", "-"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_network_stability():
    try:
        subprocess.run(["ping", "-c", "1", "google.com"], check=True, capture_output=True)
        if speedtest:
            st = speedtest.Speedtest()
            upload_speed = st.upload() / 1e6  # Mbps
            return upload_speed > 1.0  # Minimum 1 Mbps for upload
        return True
    except Exception:
        return False

def system_check():
    cpu_cores = psutil.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if cpu_cores < 3 or ram_gb < 4:
        raise ValueError("System requirements not met: Need at least 3 CPU cores and 4GB RAM")

    try:
        ffmpeg_version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
        ff_ver = version.parse(ffmpeg_version.split()[2])
        if ff_ver < version.parse("4.0"):
            raise ValueError("FFmpeg version too old")
    except Exception:
        raise ValueError("FFmpeg not found or invalid")

    import cv2
    cv_ver = version.parse(cv2.__version__)
    if cv_ver < version.parse("4.0"):
        raise ValueError("OpenCV version too old")

# Settings and Presets
SETTINGS_FILE = "settings.json"
PRESETS_FILE = "presets.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}  # Default settings

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def load_presets():
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_presets(presets):
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f)

# Update Checker
def check_for_updates(channel="Stable"):
    try:
        response = requests.get("https://api.github.com/repos/username/repo/releases")
        releases = response.json()
        latest = max(releases, key=lambda r: version.parse(r['tag_name']))
        if version.parse(latest['tag_name']) > version.parse("10.1.0") and channel in latest['prerelease']:
            return latest['zipball_url']
        return None
    except Exception:
        return None

def auto_update():
    update_url = check_for_updates()
    if update_url:
        # Download and extract
        response = requests.get(update_url)
        with open("update.zip", "wb") as f:
            f.write(response.content)
        import zipfile
        with zipfile.ZipFile("update.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("update.zip")
        # Restart
        os.execl(sys.executable, sys.executable, *sys.argv)

update_thread = threading.Thread(target=auto_update, daemon=True)
update_thread.start()

schedule.every(24).hours.do(auto_update)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Cleanup
temp_dir = tempfile.mkdtemp()

def cleanup():
    shutil.rmtree(temp_dir, ignore_errors=True)

atexit.register(cleanup)