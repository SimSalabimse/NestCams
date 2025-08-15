import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import time
import threading
import queue
import uuid
import json
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import subprocess
import logging
import requests
from datetime import datetime
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from packaging import version
import psutil
import schedule
import atexit
import platform
import shutil
from pathlib import Path

try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network checks limited.")

VERSION = "10.0.0"
UPDATE_CHANNELS = ["Stable", "Beta"]

# Logging setup
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / 'app_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VideoProcessorConfig:
    """Dataclass-like config for settings."""
    def __init__(self):
        self.batch_size = 4
        self.worker_processes = 2
        self.motion_threshold = 3000
        self.white_threshold = 200
        self.black_threshold = 50
        self.clip_limit = 1.0
        self.saturation_multiplier = 1.1
        self.music_volume = 1.0
        self.output_resolution = (1920, 1080)
        self.music_paths = {"default": None, 60: None, 720: None, 3600: None}
        self.output_dir = None
        self.custom_ffmpeg_args = None
        self.watermark_text = None
        self.update_channel = "Stable"
        self.ffmpeg_encoder = "libx264"  # Default software encoder
        self.use_hwaccel = False

class VideoProcessorApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        self.root.resizable(True, True)
        self.root.geometry("800x600")

        self.config = VideoProcessorConfig()
        self.system_info = self.detect_system()
        self.adjust_system_specs()

        if not self.check_ffmpeg():
            messagebox.showerror("Error", "FFmpeg not found. Please install FFmpeg.")
            self.root.destroy()
            return

        self.cancel_event = threading.Event()
        self.queue = queue.Queue()
        self.preview_queue = queue.Queue(maxsize=10)
        self.input_files: list[str] = []
        self.analytics_data: list[dict] = []
        self.presets: dict = {}
        self.preview_cap = None
        self.preview_running = False
        self.preview_thread = None
        self.output_format_var = tk.StringVar(value="mp4")
        self.theme_var = tk.StringVar(value="Dark")

        self.setup_gui()
        self.load_settings()
        self.load_presets()
        self.check_for_updates()
        self.root.after(50, self.process_queue)
        self.root.after(33, self.update_preview)
        atexit.register(self.cleanup)

    def detect_system(self) -> dict:
        """Detect OS, architecture, and hardware capabilities."""
        system_info = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=False) or 1,
            "cpu_threads": psutil.cpu_count(logical=True) or 1,
            "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
            "ram_available_gb": psutil.virtual_memory().available / (1024 ** 3),
            "disk_free_gb": shutil.disk_usage(Path(__file__).parent).free / (1024 ** 3),
            "gpu_available": False,
            "gpu_type": None
        }

        # GPU detection
        if system_info["os"] == "Darwin" and system_info["architecture"] == "arm64":
            # M-series Macs support VideoToolbox
            system_info["gpu_available"] = True
            system_info["gpu_type"] = "Apple VideoToolbox"
            self.config.ffmpeg_encoder = "h264_videotoolbox"
            self.config.use_hwaccel = True
        elif system_info["os"] == "Windows":
            # Check for NVIDIA GPU (NVENC)
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode == 0:
                    system_info["gpu_available"] = True
                    system_info["gpu_type"] = "NVIDIA NVENC"
                    self.config.ffmpeg_encoder = "h264_nvenc"
                    self.config.use_hwaccel = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Check for AMD GPU (AMF, less reliable detection)
                try:
                    result = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True)
                    if "cuda" in result.stdout or "dxva2" in result.stdout:
                        system_info["gpu_available"] = True
                        system_info["gpu_type"] = "AMD AMF or other"
                        self.config.ffmpeg_encoder = "h264_amf"
                        self.config.use_hwaccel = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

        logging.info(f"System detected: OS={system_info['os']}, Arch={system_info['architecture']}, "
                     f"CPU={system_info['cpu_cores']}/{system_info['cpu_threads']}, "
                     f"RAM={system_info['ram_total_gb']:.2f}GB (Avail={system_info['ram_available_gb']:.2f}GB), "
                     f"DiskFree={system_info['disk_free_gb']:.2f}GB, "
                     f"GPU={system_info['gpu_type'] or 'None'}")
        return system_info

    def adjust_system_specs(self) -> None:
        """Adjust configuration based on system specs and OS."""
        cpu_cores = self.system_info["cpu_cores"]
        cpu_threads = self.system_info["cpu_threads"]
        ram_available_gb = self.system_info["ram_available_gb"]
        disk_free_gb = self.system_info["disk_free_gb"]
        os_type = self.system_info["os"]

        # Adjust worker processes
        if os_type == "Darwin" and self.system_info["architecture"] == "arm64":
            # M-series Macs: Use efficiency cores for I/O, fewer processes
            self.config.worker_processes = min(cpu_cores, max(2, cpu_threads // 2))
        else:
            # Windows: Use more processes for high-core CPUs
            self.config.worker_processes = min(cpu_threads, max(4, cpu_cores))

        # Adjust batch size based on available RAM and disk space
        if ram_available_gb < 4 or disk_free_gb < 10:
            self.config.batch_size = 1
            self.config.worker_processes = min(self.config.worker_processes, 1)
        elif ram_available_gb < 8 or disk_free_gb < 20:
            self.config.batch_size = 2
            self.config.worker_processes = min(self.config.worker_processes, 2)
        elif ram_available_gb < 16 or disk_free_gb < 50:
            self.config.batch_size = 4
        else:
            self.config.batch_size = 8

        # Optimize OpenCV threading
        cv2.setNumThreads(self.config.worker_processes)

        logging.info(f"Optimized config: Batch={self.config.batch_size}, Workers={self.config.worker_processes}, "
                     f"FFmpegEncoder={self.config.ffmpeg_encoder}, HWAccel={self.config.use_hwaccel}")

    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed and supports required codecs."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
            version_info = result.stdout
            if self.config.use_hwaccel:
                if self.system_info["os"] == "Darwin" and "h264_videotoolbox" not in version_info:
                    logging.warning("FFmpeg does not support VideoToolbox. Falling back to software encoding.")
                    self.config.ffmpeg_encoder = "libx264"
                    self.config.use_hwaccel = False
                elif self.system_info["os"] == "Windows" and self.config.ffmpeg_encoder not in version_info:
                    logging.warning(f"FFmpeg does not support {self.config.ffmpeg_encoder}. Falling back to software encoding.")
                    self.config.ffmpeg_encoder = "libx264"
                    self.config.use_hwaccel = False
            logging.info("FFmpeg check passed")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"FFmpeg check failed: {str(e)}")
            return False

    def setup_gui(self) -> None:
        theme_frame = ctk.CTkFrame(self.root)
        theme_frame.pack(pady=5)
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(theme_frame, variable=self.theme_var, values=["Light", "Dark"], command=self.toggle_theme).pack(side=tk.LEFT)

        self.label = ctk.CTkLabel(self.root, text="Select Input Video(s)")
        self.label.pack(pady=10)

        self.generate_60s = tk.BooleanVar(value=True)
        self.switch_60s = ctk.CTkSwitch(self.root, text="Generate 60s Video", variable=self.generate_60s)
        self.switch_60s.pack(pady=5)

        self.generate_12min = tk.BooleanVar(value=True)
        self.switch_12min = ctk.CTkSwitch(self.root, text="Generate 12min Video", variable=self.generate_12min)
        self.switch_12min.pack(pady=5)

        self.generate_1h = tk.BooleanVar(value=True)
        self.switch_1h = ctk.CTkSwitch(self.root, text="Generate 1h Video", variable=self.generate_1h)
        self.switch_1h.pack(pady=5)

        custom_frame = ctk.CTkFrame(self.root)
        custom_frame.pack(pady=5)
        ctk.CTkLabel(custom_frame, text="Custom Duration (s):").pack(side=tk.LEFT, padx=5)
        self.custom_duration_entry = ctk.CTkEntry(custom_frame, placeholder_text="e.g., 120")
        self.custom_duration_entry.pack(side=tk.LEFT, padx=5)

        format_frame = ctk.CTkFrame(self.root)
        format_frame.pack(pady=5)
        ctk.CTkLabel(format_frame, text="Output Format:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(format_frame, variable=self.output_format_var, values=["mp4", "avi", "mkv", "mov", "wmv"]).pack(side=tk.LEFT)

        self.settings_button = ctk.CTkButton(self.root, text="Settings & Preview", command=self.open_settings)
        self.settings_button.pack(pady=5)

        self.browse_button = ctk.CTkButton(self.root, text="Browse", command=self.browse_files)
        self.browse_button.pack(pady=5)

        self.start_button = ctk.CTkButton(self.root, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=5)

        self.progress = ctk.CTkProgressBar(self.root, width=300)
        self.progress.pack(pady=10)
        self.progress.set(0)

        self.current_task_label = ctk.CTkLabel(self.root, text="Current Task: N/A")
        self.current_task_label.pack(pady=5)

        self.time_label = ctk.CTkLabel(self.root, text="Estimated Time Remaining: N/A")
        self.time_label.pack(pady=5)

        self.cancel_button = ctk.CTkButton(self.root, text="Cancel", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(pady=5)

        self.output_label = ctk.CTkLabel(self.root, text="Output Files:")
        self.output_label.pack(pady=10)

        self.output_frame = ctk.CTkScrollableFrame(self.root)
        self.output_frame.pack(pady=5, fill='both', expand=True)

    def toggle_theme(self, theme: str) -> None:
        ctk.set_appearance_mode(theme.lower())
        logging.info(f"Theme set to {theme}")

    def open_settings(self) -> None:
        self.settings_window = ctk.CTkToplevel(self.root)
        self.settings_window.title("Settings & Preview")
        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_settings_close)
        self.settings_window.resizable(True, True)
        self.settings_window.geometry("800x600")
        self.settings_window.transient(self.root)

        settings_frame = ctk.CTkScrollableFrame(self.settings_window)
        settings_frame.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

        ctk.CTkLabel(settings_frame, text="Motion Sensitivity").pack(pady=5)
        self.motion_slider = ctk.CTkSlider(settings_frame, from_=500, to=20000, command=self.update_settings)
        self.motion_slider.set(self.config.motion_threshold)
        self.motion_slider.pack(pady=5)
        self.motion_value_label = ctk.CTkLabel(settings_frame, text=f"Threshold: {self.config.motion_threshold}")
        self.motion_value_label.pack(pady=5)

        ctk.CTkLabel(settings_frame, text="White Threshold").pack(pady=5)
        self.white_slider = ctk.CTkSlider(settings_frame, from_=100, to=255, command=self.update_settings)
        self.white_slider.set(self.config.white_threshold)
        self.white_slider.pack(pady=5)
        self.white_value_label = ctk.CTkLabel(settings_frame, text=f"White: {self.config.white_threshold}")
        self.white_value_label.pack(pady=5)

        ctk.CTkLabel(settings_frame, text="Black Threshold").pack(pady=5)
        self.black_slider = ctk.CTkSlider(settings_frame, from_=0, to=100, command=self.update_settings)
        self.black_slider.set(self.config.black_threshold)
        self.black_slider.pack(pady=5)
        self.black_value_label = ctk.CTkLabel(settings_frame, text=f"Black: {self.config.black_threshold}")
        self.black_value_label.pack(pady=5)

        ctk.CTkLabel(settings_frame, text="CLAHE Clip Limit").pack(pady=5)
        self.clip_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=5.0, command=self.update_settings)
        self.clip_slider.set(self.config.clip_limit)
        self.clip_slider.pack(pady=5)
        self.clip_value_label = ctk.CTkLabel(settings_frame, text=f"Clip Limit: {self.config.clip_limit:.1f}")
        self.clip_value_label.pack(pady=5)

        ctk.CTkLabel(settings_frame, text="Saturation Multiplier").pack(pady=5)
        self.saturation_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=2.0, command=self.update_settings)
        self.saturation_slider.set(self.config.saturation_multiplier)
        self.saturation_slider.pack(pady=5)
        self.saturation_value_label = ctk.CTkLabel(settings_frame, text=f"Saturation: {self.config.saturation_multiplier:.1f}")
        self.saturation_value_label.pack(pady=5)

        ctk.CTkLabel(settings_frame, text="Music Volume").pack(pady=5)
        self.music_volume_slider = ctk.CTkSlider(settings_frame, from_=0.0, to=2.0, command=self.update_volume_label)
        self.music_volume_slider.set(self.config.music_volume)
        self.music_volume_slider.pack(pady=5)
        self.music_volume_label = ctk.CTkLabel(settings_frame, text=f"Volume: {self.config.music_volume:.1f}")
        self.music_volume_label.pack(pady=5)

        ctk.CTkLabel(settings_frame, text="Output Resolution").pack(pady=5)
        self.resolution_var = tk.StringVar(value="1920x1080")
        ctk.CTkOptionMenu(settings_frame, variable=self.resolution_var, values=["1280x720", "1920x1080", "2560x1440"]).pack(pady=5)

        ctk.CTkLabel(settings_frame, text="Default Music").pack(pady=5)
        self.music_label_default = ctk.CTkLabel(settings_frame, text="No music selected")
        self.music_label_default.pack(pady=5)
        ctk.CTkButton(settings_frame, text="Select Default Music", command=self.select_music_default).pack(pady=5)

        ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        ctk.CTkButton(settings_frame, text="Reset to Default", command=self.reset_to_default).pack(pady=10)

        self.preview_frame = ctk.CTkFrame(self.settings_window)
        self.preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Preview", image=ctk.CTkImage(light_image=Image.new('RGB', (200, 150), (0, 0, 0)), size=(200, 150)))
        self.preview_label.pack(pady=5)

        control_frame = ctk.CTkFrame(self.preview_frame)
        control_frame.pack(pady=5)
        self.preview_button = ctk.CTkButton(control_frame, text="Start Preview", command=self.toggle_preview, state="disabled")
        self.preview_button.pack(side=tk.LEFT, padx=5)
        self.preview_slider = ctk.CTkSlider(control_frame, from_=0, to=0, command=self.seek_preview)
        self.preview_slider.pack(side=tk.LEFT, padx=5)

        if self.input_files:
            self.initialize_preview()

    def cancel_processing(self) -> None:
        """Cancel ongoing video processing."""
        self.cancel_event.set()
        logging.info("Cancel requested")

    def load_settings(self) -> None:
        """Load settings from a JSON file."""
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.config.motion_threshold = int(settings.get("motion_threshold", 3000))
                self.config.white_threshold = int(settings.get("white_threshold", 200))
                self.config.black_threshold = int(settings.get("black_threshold", 50))
                self.config.clip_limit = float(settings.get("clip_limit", 1.0))
                self.config.saturation_multiplier = float(settings.get("saturation_multiplier", 1.1))
                self.config.music_volume = float(settings.get("music_volume", 1.0))
                self.config.output_dir = settings.get("output_dir")
                self.config.custom_ffmpeg_args = settings.get("custom_ffmpeg_args")
                self.config.watermark_text = settings.get("watermark_text")
                self.config.update_channel = settings.get("update_channel", "Stable")
                resolution_str = settings.get("output_resolution", "1920x1080")
                self.config.output_resolution = tuple(map(int, resolution_str.split('x')))
                loaded_music_paths = settings.get("music_paths", {})
                for key in self.config.music_paths:
                    if str(key) in loaded_music_paths:
                        self.config.music_paths[key] = loaded_music_paths[str(key)]
            logging.info("Loaded settings from settings.json")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            logging.warning("Could not load settings, using defaults")

    def save_settings(self) -> None:
        """Save current settings to a JSON file."""
        self.config.motion_threshold = int(self.motion_slider.get())
        self.config.white_threshold = int(self.white_slider.get())
        self.config.black_threshold = int(self.black_slider.get())
        self.config.clip_limit = float(self.clip_slider.get())
        self.config.saturation_multiplier = float(self.saturation_slider.get())
        self.config.music_volume = float(self.music_volume_slider.get())
        resolution_str = self.resolution_var.get()
        self.config.output_resolution = tuple(map(int, resolution_str.split('x')))
        settings = {
            "motion_threshold": self.config.motion_threshold,
            "white_threshold": self.config.white_threshold,
            "black_threshold": self.config.black_threshold,
            "clip_limit": self.config.clip_limit,
            "saturation_multiplier": self.config.saturation_multiplier,
            "music_volume": self.config.music_volume,
            "music_paths": {str(k): v for k, v in self.config.music_paths.items()},
            "output_dir": self.config.output_dir,
            "custom_ffmpeg_args": self.config.custom_ffmpeg_args,
            "watermark_text": self.config.watermark_text,
            "update_channel": self.config.update_channel,
            "output_resolution": resolution_str
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        self.on_settings_close()
        logging.info("Saved settings")

    def load_presets(self) -> None:
        """Load preset configurations from a JSON file."""
        try:
            with open("presets.json", "r") as f:
                self.presets = json.load(f)
            logging.info("Loaded presets from presets.json")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.info("No presets found")
            self.presets = {}

    def save_preset(self) -> None:
        """Save current settings as a preset."""
        preset_name = self.preset_name_entry.get() if hasattr(self, 'preset_name_entry') else "default"
        if preset_name:
            self.presets[preset_name] = {
                "motion_threshold": int(self.motion_slider.get()),
                "white_threshold": int(self.white_slider.get()),
                "black_threshold": int(self.black_slider.get()),
                "clip_limit": float(self.clip_slider.get()),
                "saturation_multiplier": float(self.saturation_slider.get())
            }
            with open("presets.json", "w") as f:
                json.dump(self.presets, f)
            if hasattr(self, 'preset_combobox'):
                self.preset_combobox.configure(values=list(self.presets.keys()))
            logging.info(f"Saved preset: {preset_name}")

    def check_for_updates(self) -> None:
        """Check for software updates based on the selected channel."""
        try:
            channel = self.config.update_channel
            url = f"https://raw.githubusercontent.com/SimSalabimse/NestCams/main/{channel}_version.txt"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            latest_version_str = response.text.strip()
            if not any(char.isdigit() for char in latest_version_str):
                raise ValueError(f"Invalid version: '{latest_version_str}'")
            current_version = version.parse(VERSION)
            latest_version = version.parse(latest_version_str)
            if latest_version > current_version:
                self.root.after(0, lambda: messagebox.showinfo(
                    "Update Available",
                    f"Version {latest_version_str} is available for {channel} channel! Please restart to update."
                ))
                logging.info(f"Update available for {channel}: {latest_version_str}")
            else:
                logging.info(f"No update available. Current: {VERSION}, Latest: {latest_version_str}")
        except (requests.RequestException, ValueError) as e:
            logging.error(f"Failed to check updates: {str(e)}")

    def process_queue(self) -> None:
        """Process messages from the queue for UI updates."""
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)

    def handle_message(self, message: tuple) -> None:
        """Handle different types of queue messages."""
        msg_type, *args = message
        if msg_type == "task_start":
            task_name, progress = args
            self.current_task_label.configure(text=f"Current Task: {task_name}")
            self.progress.set(progress / 100)
            self.time_label.configure(text="Estimating time...")
            logging.info(f"Task started: {task_name}")
        elif msg_type == "progress":
            progress_value, current, total, remaining = args
            progress_value = min(max(progress_value, 0), 100)
            self.progress.set(progress_value / 100)
            remaining_min = remaining / 60 if remaining > 0 else 0
            self.time_label.configure(text=f"Est. Time Remaining: {remaining_min:.2f} min ({progress_value:.2f}% complete)")
        # ... (handle other message types similarly)

    def update_preview(self) -> None:
        """Update the preview window with the latest frame."""
        if not self.preview_running or not self.preview_cap or not self.preview_cap.isOpened():
            self.root.after(33, self.update_preview)
            return
        ret, frame = self.preview_cap.read()
        if ret:
            frame = cv2.resize(frame, (200, 150))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            ctk_image = ctk.CTkImage(light_image=image, size=(200, 150))
            self.preview_label.configure(image=ctk_image)
            self.preview_label.image = ctk_image
        self.root.after(33, self.update_preview)

    def cleanup(self) -> None:
        """Clean up resources on application exit."""
        if self.preview_running:
            self.stop_preview()
        if self.preview_cap:
            self.preview_cap.release()
        logging.info("Application cleanup completed")

    def on_settings_close(self) -> None:
        """Handle settings window closure."""
        if self.preview_running:
            self.stop_preview()
        if hasattr(self, 'settings_window'):
            self.settings_window.destroy()
        logging.info("Settings window closed")

    def update_settings(self, value: float) -> None:
        """Update settings values from sliders."""
        self.config.motion_threshold = int(self.motion_slider.get())
        self.config.white_threshold = int(self.white_slider.get())
        self.config.black_threshold = int(self.black_slider.get())
        self.config.clip_limit = float(self.clip_slider.get())
        self.config.saturation_multiplier = float(self.saturation_slider.get())
        self.motion_value_label.configure(text=f"Threshold: {self.config.motion_threshold}")
        self.white_value_label.configure(text=f"White: {self.config.white_threshold}")
        self.black_value_label.configure(text=f"Black: {self.config.black_threshold}")
        self.clip_value_label.configure(text=f"Clip Limit: {self.config.clip_limit:.1f}")
        self.saturation_value_label.configure(text=f"Saturation: {self.config.saturation_multiplier:.1f}")
        logging.info(f"Updated settings: Motion={self.config.motion_threshold}, White={self.config.white_threshold}, Black={self.config.black_threshold}, Clip={self.config.clip_limit}, Saturation={self.config.saturation_multiplier}")

    def update_volume_label(self, value: float) -> None:
        """Update music volume label."""
        self.config.music_volume = float(value)
        self.music_volume_label.configure(text=f"Volume: {self.config.music_volume:.1f}")

    def reset_to_default(self) -> None:
        """Reset settings to default values."""
        self.motion_slider.set(3000)
        self.white_slider.set(200)
        self.black_slider.set(50)
        self.clip_slider.set(1.0)
        self.saturation_slider.set(1.1)
        self.music_volume_slider.set(1.0)
        self.resolution_var.set("1920x1080")
        self.config.output_dir = None
        self.config.custom_ffmpeg_args = None
        self.config.watermark_text = None
        self.config.update_channel = "Stable"
        self.update_settings(0)
        self.update_volume_label(1.0)
        logging.info("Reset settings to default values")

    def initialize_preview(self) -> None:
        """Initialize video preview."""
        if self.input_files and not self.preview_cap:
            self.preview_cap = cv2.VideoCapture(self.input_files[0])
            if self.preview_cap.isOpened():
                total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.preview_slider.configure(from_=0, to=total_frames - 1)
                self.preview_button.configure(state="normal")
                logging.info("Preview initialized")

    def toggle_preview(self) -> None:
        """Start or stop video preview."""
        if self.preview_running:
            self.stop_preview()
            self.preview_button.configure(text="Start Preview")
        else:
            self.start_preview()
            self.preview_button.configure(text="Stop Preview")

    def start_preview(self) -> None:
        """Start video preview."""
        if self.preview_cap and not self.preview_running:
            self.preview_running = True
            self.preview_thread = threading.Thread(target=self.run_preview)
            self.preview_thread.start()
            logging.info("Preview started")

    def stop_preview(self) -> None:
        """Stop video preview."""
        self.preview_running = False
        if self.preview_thread:
            self.preview_thread.join()
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logging.info("Preview stopped")

    def seek_preview(self, frame: float) -> None:
        """Seek to a specific frame in the preview."""
        if self.preview_cap:
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            logging.info(f"Preview seek to frame {int(frame)}")

    def run_preview(self) -> None:
        """Run preview loop."""
        while self.preview_running and self.preview_cap.isOpened():
            ret, frame = self.preview_cap.read()
            if not ret:
                self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (200, 150))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            self.preview_queue.put(image)
            time.sleep(1 / 30)  # ~30 FPS

    def select_music_default(self) -> None:
        """Select default background music."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.config.music_paths["default"] = path
            self.music_label_default.configure(text=os.path.basename(path))
        else:
            self.config.music_paths["default"] = None
            self.music_label_default.configure(text="No music selected")
        logging.info(f"Selected default music: {path}")

    def disable_processing_ui(self) -> None:
        """Disable UI elements during processing."""
        self.switch_60s.configure(state="disabled")
        self.switch_12min.configure(state="disabled")
        self.switch_1h.configure(state="disabled")
        self.custom_duration_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        logging.info("Processing UI disabled")

    def show_analytics_dashboard(self) -> None:
        """Display analytics after processing."""
        analytics_window = ctk.CTkToplevel(self.root)
        analytics_window.title("Processing Analytics")
        analytics_window.geometry("600x400")
        analytics_window.transient(self.root)

        ctk.CTkLabel(analytics_window, text="Processing Analytics").pack(pady=10)
        frame = ctk.CTkScrollableFrame(analytics_window)
        frame.pack(pady=5, fill='both', expand=True)

        for data in self.analytics_data:
            file_frame = ctk.CTkFrame(frame)
            file_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(file_frame, text=f"File: {data['file']}").pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(file_frame, text=f"Duration: {data['duration']}s").pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(file_frame, text=f"Frames: {data['frames_processed']}").pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(file_frame, text=f"Motion Events: {data['motion_events']}").pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(file_frame, text=f"Time: {data['processing_time']:.2f}s").pack(side=tk.LEFT, padx=5)
        logging.info("Displayed analytics dashboard")

    def get_output_path(self, input_file: str, duration: int) -> str:
        """Generate output file path."""
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}_{duration}s.{self.output_format_var.get()}"
        if self.config.output_dir:
            output_file = str(Path(self.config.output_dir) / Path(output_file).name)
        return output_file

    def reset_ui(self) -> None:
        """Reset UI elements to initial state."""
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.switch_1h.configure(state="normal")
        self.custom_duration_entry.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.cancel_button.configure(state="disabled")
        logging.info("UI reset to initial state")

    def browse_files(self) -> None:
        self.input_files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv")])
        if self.input_files:
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
                self.initialize_preview()
            logging.info(f"Selected files: {self.input_files}")

    def start_processing(self) -> None:
        if not self.input_files:
            messagebox.showwarning("Warning", "No files selected.")
            return
        durations = []
        if self.generate_60s.get():
            durations.append(60)
        if self.generate_12min.get():
            durations.append(720)
        if self.generate_1h.get():
            durations.append(3600)
        custom = self.custom_duration_entry.get()
        if custom:
            try:
                durations.append(int(custom))
            except ValueError:
                messagebox.showwarning("Warning", "Invalid custom duration.")
                return
        if not durations:
            messagebox.showwarning("Warning", "Select at least one duration.")
            return

        self.disable_processing_ui()
        self.cancel_event.clear()
        threading.Thread(target=self.process_videos, args=(durations,)).start()

    def process_videos(self, durations: list[int]) -> None:
        try:
            total_tasks = len(self.input_files) * (len(durations) + 1)
            task_count = 0
            for input_file in self.input_files:
                if self.cancel_event.is_set():
                    break
                task_count = self.process_single_video(input_file, durations, task_count, total_tasks)
            if self.analytics_data and not self.cancel_event.is_set():
                self.root.after(0, self.show_analytics_dashboard)
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            self.queue.put(("canceled", str(e)))
        finally:
            self.root.after(0, self.reset_ui)

    def process_single_video(self, input_file: str, durations: list[int], task_count: int, total_tasks: int) -> int:
        self.queue.put(("task_start", f"Motion Detection - {os.path.basename(input_file)}", task_count / total_tasks * 100))
        task_count += 1
        selected_indices = self.get_selected_indices(input_file)
        if selected_indices is None or not selected_indices:
            self.queue.put(("canceled", "Motion detection failed or canceled"))
            return task_count

        with ProcessPoolExecutor(max_workers=self.config.worker_processes) as executor:
            futures = []
            for duration in durations:
                if self.cancel_event.is_set():
                    break
                output_path = self.get_output_path(input_file, duration)
                future = executor.submit(self.generate_output_video, input_file, output_path, duration, selected_indices)
                futures.append((duration, future, task_count))
                task_count += 1
            for duration, future, t_count in futures:
                if self.cancel_event.is_set():
                    break
                self.queue.put(("task_start", f"Generate {duration}s - {os.path.basename(input_file)}", t_count / total_tasks * 100))
                result = future.result()
                if result[0]:
                    self.queue.put(("canceled", result[0]))
                else:
                    self.analytics_data.append(result[1])
        return task_count

    def get_selected_indices(self, input_path: str) -> list[int] | None:
        start_time = time.time()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error(f"Cannot open {input_path}")
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prev_frame = None
        selected = []
        for idx in range(total_frames):
            if self.cancel_event.is_set():
                cap.release()
                return None
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (640, 360))
            if prev_frame is not None:
                score = self.compute_motion_score(prev_frame, frame_resized)
                if score > self.config.motion_threshold and not self.is_white_or_black_frame(frame_resized):
                    selected.append(idx)
            prev_frame = frame_resized
            if idx % 100 == 0:
                progress = (idx / total_frames) * 100
                self.queue.put(("progress", progress, idx, total_frames, (total_frames - idx) / (idx / (time.time() - start_time) if idx > 0 else 1)))
        cap.release()
        return selected

    def compute_motion_score(self, prev: np.ndarray, curr: np.ndarray) -> int:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        diff = np.abs(prev_gray - curr_gray) > 30
        return np.sum(diff)

    def is_white_or_black_frame(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        return mean > self.config.white_threshold or mean < self.config.black_threshold

    def generate_output_video(self, input_path: str, output_path: str, duration: int, selected_indices: list[int]) -> tuple[str | None, dict]:
        start_time = time.time()
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0 or fps <= 0:
            return "Invalid video", {}

        rotate = duration <= 60
        target_frames = int(duration * fps)
        final_indices = selected_indices[::max(1, len(selected_indices) // target_frames)] if len(selected_indices) > target_frames else selected_indices

        with tempfile.TemporaryDirectory() as temp_dir:
            with ProcessPoolExecutor(max_workers=self.config.worker_processes) as executor:
                batches = [final_indices[i:i + self.config.batch_size] for i in range(0, len(final_indices), self.config.batch_size)]
                futures = [executor.submit(self.process_frame_batch, input_path, batch, temp_dir, rotate) for batch in batches]
                frame_counter = 0
                for future in as_completed(futures):
                    frame_counter += len(future.result())

            if frame_counter == 0:
                return "No frames processed", {}

            new_fps = frame_counter / duration
            temp_video = str(Path(temp_dir) / "temp.mp4")
            cmd = ['ffmpeg', '-framerate', str(new_fps), '-i', str(Path(temp_dir) / 'frame_%04d.jpg'), '-c:v', self.config.ffmpeg_encoder, '-pix_fmt', 'yuv420p', '-r', str(new_fps), '-y', temp_video]
            if self.config.use_hwaccel:
                cmd.insert(1, '-hwaccel')
                cmd.insert(2, 'auto')
            if self.config.watermark_text:
                cmd.extend(['-vf', f'drawtext=text={self.config.watermark_text}:fontcolor=white:fontsize=24:x=10:y=10'])
            if self.config.custom_ffmpeg_args:
                cmd.extend(self.config.custom_ffmpeg_args)
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg error: {e.stderr.decode()}")
                return f"FFmpeg error: {e.stderr.decode()}", {}

            music_path = self.config.music_paths.get(duration, self.config.music_paths["default"])
            if music_path and os.path.exists(music_path):
                cmd = ['ffmpeg', '-i', temp_video, '-i', music_path, '-filter_complex', f"[1:a]volume={self.config.music_volume}[a]", '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path]
            else:
                cmd = ['ffmpeg', '-i', temp_video, '-f', 'lavfi', '-i', 'anullsrc', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg error: {e.stderr.decode()}")
                return f"FFmpeg error: {e.stderr.decode()}", {}

        proc_time = time.time() - start_time
        data = {"file": os.path.basename(input_path), "duration": duration, "frames_processed": frame_counter, "motion_events": len(selected_indices), "processing_time": proc_time}
        return None, data

    def process_frame_batch(self, input_path: str, indices: list[int], temp_dir: str, rotate: bool) -> list[int]:
        cap = cv2.VideoCapture(input_path)
        processed = []
        for order, idx in enumerate(indices):
            if self.cancel_event.is_set():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            norm_frame = self.normalize_frame(frame)
            if norm_frame is None:
                continue
            if rotate:
                norm_frame = cv2.rotate(norm_frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(str(Path(temp_dir) / f"frame_{order:04d}.jpg"), norm_frame)
            processed.append(order)
        cap.release()
        return processed

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray | None:
        try:
            frame = cv2.resize(frame, self.config.output_resolution)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.config.clip_limit)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s * self.config.saturation_multiplier, 0, 255).astype(np.uint8)
            hsv = cv2.merge((h, s, v))
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            logging.error(f"Normalize error: {e}")
            return None

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()