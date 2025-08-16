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
import schedule
from multiprocessing import Pool, Event as MpEvent
from concurrent.futures import ThreadPoolExecutor
from packaging import version
import functools
import psutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except ImportError:
    TkinterDnD = None
    DND_FILES = None
    logging.warning("tkinterdnd2 not available, drag-and-drop disabled")

try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network stability checks limited")

VERSION = "10.0.0"
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

def log_session(message):
    """Log a message to the session log file."""
    session_logger.info(message)

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

class VideoProcessorApp:
    """Main application class for Bird Box Video Processor."""
    def __init__(self, root):
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        log_session("Application started")
        self.root.resizable(True, True)
        self.root.geometry("900x700")

        # Initialize instance variables first
        self.batch_size = 4
        self.worker_processes = 2
        self.output_resolution = (1920, 1080)
        self.cancel_event = MpEvent()
        self.pause_event = MpEvent()
        self.paused = False
        self.thread_lock = threading.Lock()
        self.motion_threshold = 3000
        self.white_threshold = 200
        self.black_threshold = 50
        self.clip_limit = 1.0
        self.saturation_multiplier = 1.1
        self.music_volume = 1.0
        self.output_dir = None
        self.custom_ffmpeg_args = None
        self.watermark_text = None
        self.preview_running = False
        self.preview_thread = None
        self.preview_cap = None
        self.update_channel = "Stable"
        self.music_paths = {"default": None, 60: None, 720: None, 3600: None}
        self.analytics_data = []
        self.presets = {}
        self.progress_rows = {}
        self.input_files = []
        self.queue = queue.Queue()
        self.preview_queue = queue.Queue(maxsize=5)
        self.preview_image = None
        self.blank_ctk_image = ctk.CTkImage(
            light_image=Image.new('RGB', (200, 150), (0, 0, 0)),
            dark_image=Image.new('RGB', (200, 150), (0, 0, 0)),
            size=(200, 150)
        )

        # Check system specs
        self.check_system_specs()

        # Load settings and presets
        self.load_settings()
        self.load_presets()

        # Setup GUI tabs
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(pady=10, padx=10, fill="both", expand=True)
        self.main_tab = self.tabview.add("Main")
        self.settings_tab = self.tabview.add("Settings")
        self.music_tab = self.tabview.add("Music")
        self.advanced_tab = self.tabview.add("Advanced")
        self.help_tab = self.tabview.add("Help")
        self.setup_main_tab()
        self.setup_settings_tab()
        self.setup_music_tab()
        self.setup_advanced_tab()
        self.setup_help_tab()

        # Setup drag-and-drop
        if TkinterDnD:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.browse_files())
        self.root.bind('<Control-s>', lambda e: self.start_processing() if self.start_button.cget("state") == "normal" else None)
        self.root.bind('<Control-c>', lambda e: self.cancel_processing() if self.cancel_button.cget("state") == "normal" else None)

        # Periodic tasks
        self.root.after(50, self.process_queue)
        self.root.after(33, self.update_preview)
        self.check_for_updates()

    def check_system_specs(self):
        """Adjust processing parameters based on system specs."""
        cpu_cores = os.cpu_count() or 1
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_ram_gb < 8:
            self.batch_size = max(1, cpu_cores // 2)
            self.worker_processes = 1
        elif total_ram_gb < 16:
            self.batch_size = max(2, cpu_cores // 2)
            self.worker_processes = min(2, cpu_cores)
        elif total_ram_gb < 32:
            self.batch_size = max(4, cpu_cores)
            self.worker_processes = min(4, cpu_cores)
        else:
            self.batch_size = max(8, cpu_cores * 2)
            self.worker_processes = min(8, cpu_cores)
        logging.info(f"System specs: CPU cores={cpu_cores}, RAM={total_ram_gb:.2f} GB")
        log_session(f"Configured: Batch size={self.batch_size}, Workers={self.worker_processes}")

    def setup_main_tab(self):
        """Setup Main tab UI."""
        self.label = ctk.CTkLabel(self.main_tab, text="Select Input Video(s) or Drag & Drop")
        self.label.pack(pady=10)
        self.generate_60s = tk.BooleanVar(value=True)
        self.switch_60s = ctk.CTkSwitch(self.main_tab, text="Generate 60s Video", variable=self.generate_60s)
        self.switch_60s.pack(pady=5)
        ToolTip(self.switch_60s, "Generate a 60-second condensed video (vertical orientation)")
        self.generate_12min = tk.BooleanVar(value=True)
        self.switch_12min = ctk.CTkSwitch(self.main_tab, text="Generate 12min Video", variable=self.generate_12min)
        self.switch_12min.pack(pady=5)
        ToolTip(self.switch_12min, "Generate a 12-minute condensed video")
        self.generate_1h = tk.BooleanVar(value=True)
        self.switch_1h = ctk.CTkSwitch(self.main_tab, text="Generate 1h Video", variable=self.generate_1h)
        self.switch_1h.pack(pady=5)
        ToolTip(self.switch_1h, "Generate a 1-hour condensed video")
        custom_frame = ctk.CTkFrame(self.main_tab)
        custom_frame.pack(pady=5)
        ctk.CTkLabel(custom_frame, text="Custom Duration (seconds):").pack(side=tk.LEFT, padx=5)
        self.custom_duration_entry = ctk.CTkEntry(custom_frame, placeholder_text="e.g., 120")
        self.custom_duration_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(self.custom_duration_entry, "Enter custom video duration in seconds")
        self.output_format_var = tk.StringVar(value="mp4")
        format_frame = ctk.CTkFrame(self.main_tab)
        format_frame.pack(pady=5)
        ctk.CTkLabel(format_frame, text="Output Format:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(format_frame, variable=self.output_format_var, values=["mp4", "avi", "mkv", "mov", "wmv"]).pack(side=tk.LEFT)
        ToolTip(format_frame, "Select output video format")
        self.browse_button = ctk.CTkButton(self.main_tab, text="Browse", command=self.browse_files)
        self.browse_button.pack(pady=5)
        ToolTip(self.browse_button, "Browse for video files (Ctrl+O)")
        self.start_button = ctk.CTkButton(self.main_tab, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=5)
        ToolTip(self.start_button, "Start processing videos (Ctrl+S)")
        self.pause_button = ctk.CTkButton(self.main_tab, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(pady=5)
        ToolTip(self.pause_button, "Pause/Resume processing")
        self.cancel_button = ctk.CTkButton(self.main_tab, text="Cancel", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(pady=5)
        ToolTip(self.cancel_button, "Cancel processing (Ctrl+C)")
        self.progress_frame = ctk.CTkScrollableFrame(self.main_tab)
        self.progress_frame.pack(pady=10, fill='both', expand=True)
        ToolTip(self.progress_frame, "Per-file processing progress")
        self.output_label = ctk.CTkLabel(self.main_tab, text="Output Files:")
        self.output_label.pack(pady=10)
        self.output_frame = ctk.CTkScrollableFrame(self.main_tab)
        self.output_frame.pack(pady=5, fill='x')

    def setup_settings_tab(self):
        """Setup Settings tab UI."""
        self.settings_frame = ctk.CTkScrollableFrame(self.settings_tab)
        self.settings_frame.pack(padx=10, pady=10, fill='both', expand=True)
        ctk.CTkLabel(self.settings_frame, text="Motion Sensitivity").pack(pady=5)
        self.motion_slider = ctk.CTkSlider(self.settings_frame, from_=500, to=20000, number_of_steps=195, command=self.update_settings)
        self.motion_slider.set(self.motion_threshold)
        self.motion_slider.pack(pady=5)
        self.motion_value_label = ctk.CTkLabel(self.settings_frame, text=f"Threshold: {self.motion_threshold}")
        self.motion_value_label.pack(pady=5)
        ToolTip(self.motion_slider, "Higher value means less sensitive to motion")
        ctk.CTkLabel(self.settings_frame, text="White Threshold").pack(pady=2)
        self.white_slider = ctk.CTkSlider(self.settings_frame, from_=100, to=255, number_of_steps=155, command=self.update_settings)
        self.white_slider.set(self.white_threshold)
        self.white_slider.pack(pady=2)
        self.white_value_label = ctk.CTkLabel(self.settings_frame, text=f"White: {self.white_threshold}")
        self.white_value_label.pack(pady=2)
        ToolTip(self.white_slider, "Threshold for detecting overly white frames")
        ctk.CTkLabel(self.settings_frame, text="Black Threshold").pack(pady=2)
        self.black_slider = ctk.CTkSlider(self.settings_frame, from_=0, to=100, number_of_steps=100, command=self.update_settings)
        self.black_slider.set(self.black_threshold)
        self.black_slider.pack(pady=2)
        self.black_value_label = ctk.CTkLabel(self.settings_frame, text=f"Black: {self.black_threshold}")
        self.black_value_label.pack(pady=2)
        ToolTip(self.black_slider, "Threshold for detecting overly black frames")
        ctk.CTkLabel(self.settings_frame, text="CLAHE Clip Limit").pack(pady=2)
        self.clip_slider = ctk.CTkSlider(self.settings_frame, from_=0.2, to=5.0, number_of_steps=96, command=self.update_settings)
        self.clip_slider.set(self.clip_limit)
        self.clip_slider.pack(pady=2)
        self.clip_value_label = ctk.CTkLabel(self.settings_frame, text=f"Clip Limit: {self.clip_limit:.1f}")
        self.clip_value_label.pack(pady=2)
        ToolTip(self.clip_slider, "Contrast enhancement limit")
        ctk.CTkLabel(self.settings_frame, text="Saturation Multiplier").pack(pady=2)
        self.saturation_slider = ctk.CTkSlider(self.settings_frame, from_=0.5, to=2.0, number_of_steps=150, command=self.update_settings)
        self.saturation_slider.set(self.saturation_multiplier)
        self.saturation_slider.pack(pady=2)
        self.saturation_value_label = ctk.CTkLabel(self.settings_frame, text=f"Saturation: {self.saturation_multiplier:.1f}")
        self.saturation_value_label.pack(pady=2)
        ToolTip(self.saturation_slider, "Multiplier for color saturation")
        ctk.CTkLabel(self.settings_frame, text="Output Resolution").pack(pady=5)
        resolution_options = ["320x180", "640x360", "1280x720", "1920x1080"]
        self.resolution_var = tk.StringVar(value=f"{self.output_resolution[0]}x{self.output_resolution[1]}")
        self.resolution_menu = ctk.CTkOptionMenu(self.settings_frame, variable=self.resolution_var, values=resolution_options)
        self.resolution_menu.pack(pady=5)
        ToolTip(self.resolution_menu, "Select output video resolution")
        self.preview_frame = ctk.CTkFrame(self.settings_tab)
        self.preview_frame.pack(pady=10)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Preview (Select video first)", image=self.blank_ctk_image)
        self.preview_label.pack(pady=5)
        ToolTip(self.preview_label, "Live preview of selected video")
        control_frame = ctk.CTkFrame(self.preview_frame)
        control_frame.pack(pady=5)
        self.preview_button = ctk.CTkButton(control_frame, text="Start Preview", command=self.toggle_preview, state="disabled")
        self.preview_button.pack(side=tk.LEFT, padx=5)
        ToolTip(self.preview_button, "Start/Stop video preview")
        self.preview_slider = ctk.CTkSlider(control_frame, from_=0, to=0, number_of_steps=0, command=self.seek_preview)
        self.preview_slider.pack(side=tk.LEFT, padx=5)
        ToolTip(self.preview_slider, "Seek through video frames")

    def setup_music_tab(self):
        """Setup Music tab UI."""
        music_settings_frame = ctk.CTkFrame(self.music_tab)
        music_settings_frame.pack(pady=10, padx=10, fill='both', expand=True)
        ctk.CTkLabel(music_settings_frame, text="Music Settings").pack(pady=5)
        default_music_frame = ctk.CTkFrame(music_settings_frame)
        default_music_frame.pack(pady=2)
        ctk.CTkLabel(default_music_frame, text="Default Music:").pack(side=tk.LEFT)
        self.music_label_default = ctk.CTkLabel(default_music_frame, text="No music selected")
        self.music_label_default.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(default_music_frame, text="Select", command=self.select_music_default).pack(side=tk.LEFT)
        ToolTip(default_music_frame, "Select default background music")
        music_60s_frame = ctk.CTkFrame(music_settings_frame)
        music_60s_frame.pack(pady=2)
        ctk.CTkLabel(music_60s_frame, text="Music for 60s Video:").pack(side=tk.LEFT)
        self.music_label_60s = ctk.CTkLabel(music_60s_frame, text="No music selected")
        self.music_label_60s.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_60s_frame, text="Select", command=self.select_music_60s).pack(side=tk.LEFT)
        ToolTip(music_60s_frame, "Select music for 60s videos")
        music_12min_frame = ctk.CTkFrame(music_settings_frame)
        music_12min_frame.pack(pady=2)
        ctk.CTkLabel(music_12min_frame, text="Music for 12min Video:").pack(side=tk.LEFT)
        self.music_label_12min = ctk.CTkLabel(music_12min_frame, text="No music selected")
        self.music_label_12min.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_12min_frame, text="Select", command=self.select_music_12min).pack(side=tk.LEFT)
        ToolTip(music_12min_frame, "Select music for 12min videos")
        music_1h_frame = ctk.CTkFrame(music_settings_frame)
        music_1h_frame.pack(pady=2)
        ctk.CTkLabel(music_1h_frame, text="Music for 1h Video:").pack(side=tk.LEFT)
        self.music_label_1h = ctk.CTkLabel(music_1h_frame, text="No music selected")
        self.music_label_1h.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_1h_frame, text="Select", command=self.select_music_1h).pack(side=tk.LEFT)
        ToolTip(music_1h_frame, "Select music for 1h videos")
        volume_frame = ctk.CTkFrame(music_settings_frame)
        volume_frame.pack(pady=2)
        ctk.CTkLabel(volume_frame, text="Music Volume (0.0 - 1.0):").pack(side=tk.LEFT)
        self.music_volume_slider = ctk.CTkSlider(volume_frame, from_=0.0, to=1.0, number_of_steps=100, command=self.update_volume_label)
        self.music_volume_slider.set(self.music_volume)
        self.music_volume_slider.pack(side=tk.LEFT)
        self.volume_value_label = ctk.CTkLabel(volume_frame, text=f"{int(self.music_volume * 100)}%")
        self.volume_value_label.pack(side=tk.LEFT, padx=5)
        ToolTip(volume_frame, "Adjust music volume level")

    def setup_advanced_tab(self):
        """Setup Advanced tab UI."""
        advanced_frame = ctk.CTkScrollableFrame(self.advanced_tab)
        advanced_frame.pack(padx=10, pady=10, fill='both', expand=True)
        self.theme_var = tk.StringVar(value="Dark")
        theme_frame = ctk.CTkFrame(advanced_frame)
        theme_frame.pack(pady=5)
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(theme_frame, variable=self.theme_var, values=["Light", "Dark"], command=self.toggle_theme).pack(side=tk.LEFT)
        ToolTip(theme_frame, "Switch between light and dark themes")
        update_channel_frame = ctk.CTkFrame(advanced_frame)
        update_channel_frame.pack(pady=5)
        ctk.CTkLabel(update_channel_frame, text="Update Channel:").pack(side=tk.LEFT)
        self.update_channel_var = tk.StringVar(value=self.update_channel)
        ctk.CTkOptionMenu(update_channel_frame, variable=self.update_channel_var, values=UPDATE_CHANNELS).pack(side=tk.LEFT)
        ToolTip(update_channel_frame, "Select update channel (Stable or Beta)")
        output_dir_frame = ctk.CTkFrame(advanced_frame)
        output_dir_frame.pack(pady=5)
        ctk.CTkLabel(output_dir_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_label = ctk.CTkLabel(output_dir_frame, text=self.output_dir or "Default")
        self.output_dir_label.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(output_dir_frame, text="Browse", command=self.select_output_dir).pack(side=tk.LEFT)
        ToolTip(output_dir_frame, "Select custom output directory")
        ffmpeg_frame = ctk.CTkFrame(advanced_frame)
        ffmpeg_frame.pack(pady=5)
        ctk.CTkLabel(ffmpeg_frame, text="Custom FFmpeg Args:").pack(side=tk.LEFT)
        self.ffmpeg_entry = ctk.CTkEntry(ffmpeg_frame, placeholder_text="e.g., -vf scale=1280:720")
        self.ffmpeg_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(ffmpeg_frame, "Additional FFmpeg command-line arguments")
        watermark_frame = ctk.CTkFrame(advanced_frame)
        watermark_frame.pack(pady=5)
        ctk.CTkLabel(watermark_frame, text="Watermark Text:").pack(side=tk.LEFT)
        self.watermark_entry = ctk.CTkEntry(watermark_frame, placeholder_text="Enter watermark")
        self.watermark_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(watermark_frame, "Text to watermark on videos")
        schedule_frame = ctk.CTkFrame(advanced_frame)
        schedule_frame.pack(pady=10)
        ctk.CTkLabel(schedule_frame, text="Schedule Processing (HH:MM):").pack(side=tk.LEFT)
        self.schedule_entry = ctk.CTkEntry(schedule_frame, placeholder_text="e.g., 14:30")
        self.schedule_entry.pack(pady=2)
        ctk.CTkButton(schedule_frame, text="Set Schedule", command=self.set_schedule).pack(pady=2)
        ToolTip(schedule_frame, "Schedule daily processing time")
        preset_frame = ctk.CTkFrame(advanced_frame)
        preset_frame.pack(pady=10)
        ctk.CTkLabel(preset_frame, text="Preset Management").pack(pady=5)
        self.preset_combobox = ctk.CTkComboBox(preset_frame, values=list(self.presets.keys()))
        self.preset_combobox.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Load Preset", command=self.load_preset).pack(pady=5)
        self.preset_name_entry = ctk.CTkEntry(preset_frame, placeholder_text="Enter preset name")
        self.preset_name_entry.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Save Preset", command=self.save_preset).pack(pady=5)
        ToolTip(preset_frame, "Save/load preset configurations")
        ctk.CTkButton(advanced_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        ctk.CTkButton(advanced_frame, text="Reset to Default", command=self.reset_to_default).pack(pady=10)

    def setup_help_tab(self):
        """Setup Help tab with documentation."""
        help_text = (
            "Bird Box Video Processor Help\n\n"
            "1. Main Tab: Select videos, choose durations, start processing.\n"
            "2. Settings Tab: Adjust motion detection and enhancement parameters.\n"
            "3. Music Tab: Select background music for different video lengths.\n"
            "4. Advanced Tab: Custom FFmpeg args, watermark, scheduling, presets.\n\n"
            "Tips:\n- Drag and drop videos into the window.\n- Use Ctrl+O (browse), Ctrl+S (start), Ctrl+C (cancel).\n"
            "- Check logs in the 'log' folder.\n- Ensure client_secrets.json for YouTube upload.\n\n"
            f"Version: {VERSION}"
        )
        ctk.CTkLabel(self.help_tab, text=help_text, justify=tk.LEFT).pack(pady=10, padx=10, anchor="nw")

    def load_settings(self):
        """Load settings from JSON."""
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.motion_threshold = settings.get("motion_threshold", 3000)
                self.white_threshold = settings.get("white_threshold", 200)
                self.black_threshold = settings.get("black_threshold", 50)
                self.clip_limit = settings.get("clip_limit", 1.0)
                self.saturation_multiplier = settings.get("saturation_multiplier", 1.1)
                self.music_volume = settings.get("music_volume", 1.0)
                self.output_dir = settings.get("output_dir", None)
                self.custom_ffmpeg_args = settings.get("custom_ffmpeg_args", None)
                self.watermark_text = settings.get("watermark_text", None)
                self.update_channel = settings.get("update_channel", "Stable")
                resolution_str = settings.get("output_resolution", "1920x1080")
                self.output_resolution = tuple(map(int, resolution_str.split('x')))
                loaded_music_paths = settings.get("music_paths", {})
                for key in self.music_paths:
                    str_key = str(key)
                    if str_key in loaded_music_paths:
                        self.music_paths[key] = loaded_music_paths[str_key]
                        if key == "default":
                            self.music_label_default.configure(text=os.path.basename(self.music_paths[key]) if self.music_paths[key] else "No music selected")
                        elif key == 60:
                            self.music_label_60s.configure(text=os.path.basename(self.music_paths[key]) if self.music_paths[key] else "No music selected")
                        elif key == 720:
                            self.music_label_12min.configure(text=os.path.basename(self.music_paths[key]) if self.music_paths[key] else "No music selected")
                        elif key == 3600:
                            self.music_label_1h.configure(text=os.path.basename(self.music_paths[key]) if self.music_paths[key] else "No music selected")
            log_session("Loaded settings from settings.json")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            logging.warning("Could not load settings, using defaults")
            log_session("Could not load settings, using defaults")

    def save_settings(self):
        """Save settings to JSON."""
        self.motion_threshold = int(self.motion_slider.get())
        self.white_threshold = int(self.white_slider.get())
        self.black_threshold = int(self.black_slider.get())
        self.clip_limit = float(self.clip_slider.get())
        self.saturation_multiplier = float(self.saturation_slider.get())
        self.music_volume = self.music_volume_slider.get()
        resolution_str = self.resolution_var.get()
        self.output_resolution = tuple(map(int, resolution_str.split('x')))
        self.custom_ffmpeg_args = self.ffmpeg_entry.get().split() if self.ffmpeg_entry.get() else None
        self.watermark_text = self.watermark_entry.get() or None
        self.update_channel = self.update_channel_var.get()
        settings = {
            "motion_threshold": self.motion_threshold,
            "white_threshold": self.white_threshold,
            "black_threshold": self.black_threshold,
            "clip_limit": self.clip_limit,
            "saturation_multiplier": self.saturation_multiplier,
            "music_volume": self.music_volume,
            "music_paths": {str(k): v for k, v in self.music_paths.items()},
            "output_dir": self.output_dir,
            "custom_ffmpeg_args": self.custom_ffmpeg_args,
            "watermark_text": self.watermark_text,
            "update_channel": self.update_channel,
            "output_resolution": resolution_str
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        log_session("Saved settings")

    def load_presets(self):
        """Load presets from JSON."""
        try:
            with open("presets.json", "r") as f:
                self.presets = json.load(f)
            self.preset_combobox.configure(values=list(self.presets.keys()))
            log_session("Loaded presets")
        except (FileNotFoundError, json.JSONDecodeError):
            log_session("No presets found")
            self.presets = {}

    def save_preset(self):
        """Save current settings as preset."""
        preset_name = self.preset_name_entry.get()
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
            self.preset_combobox.configure(values=list(self.presets.keys()))
            log_session(f"Saved preset: {preset_name}")

    def load_preset(self):
        """Load selected preset."""
        preset_name = self.preset_combobox.get()
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            self.motion_slider.set(preset.get("motion_threshold", 3000))
            self.white_slider.set(preset.get("white_threshold", 200))
            self.black_slider.set(preset.get("black_threshold", 50))
            self.clip_slider.set(preset.get("clip_limit", 1.0))
            self.saturation_slider.set(preset.get("saturation_multiplier", 1.1))
            self.update_settings(0)
            log_session(f"Loaded preset: {preset_name}")

    def reset_to_default(self):
        """Reset settings to defaults."""
        self.motion_slider.set(3000)
        self.white_slider.set(200)
        self.black_slider.set(50)
        self.clip_slider.set(1.0)
        self.saturation_slider.set(1.1)
        self.resolution_var.set("1920x1080")
        self.music_volume_slider.set(1.0)
        self.output_dir = None
        self.output_dir_label.configure(text="Default")
        self.ffmpeg_entry.delete(0, tk.END)
        self.watermark_entry.delete(0, tk.END)
        self.update_channel_var.set("Stable")
        self.update_settings(0)
        self.update_volume_label(1.0)
        log_session("Reset settings to default")

    def check_for_updates(self):
        """Check for software updates."""
        try:
            channel = self.update_channel
            url = f"https://raw.githubusercontent.com/SimSalabimse/NestCams/main/{channel}_version.txt"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            latest_version_str = response.text.strip()
            current_version = version.parse(VERSION)
            latest_version = version.parse(latest_version_str)
            if latest_version > current_version:
                messagebox.showinfo("Update Available", f"Version {latest_version_str} available for {channel}! Restart to update.")
                log_session(f"Update available: {latest_version_str}")
            else:
                log_session(f"No update. Current: {VERSION}, Latest: {latest_version_str}")
        except Exception as e:
            log_session(f"Update check failed: {str(e)}")

    def toggle_theme(self, theme):
        """Toggle UI theme."""
        ctk.set_appearance_mode(theme.lower())
        log_session(f"Theme changed to {theme}")

    def update_settings(self, value):
        """Update settings from sliders."""
        self.motion_threshold = int(self.motion_slider.get())
        self.white_threshold = int(self.white_slider.get())
        self.black_threshold = int(self.black_slider.get())
        self.clip_limit = float(self.clip_slider.get())
        self.saturation_multiplier = float(self.saturation_slider.get())
        self.motion_value_label.configure(text=f"Threshold: {self.motion_threshold}")
        self.white_value_label.configure(text=f"White: {self.white_threshold}")
        self.black_value_label.configure(text=f"Black: {self.black_threshold}")
        self.clip_value_label.configure(text=f"Clip Limit: {self.clip_limit:.1f}")
        self.saturation_value_label.configure(text=f"Saturation: {self.saturation_multiplier:.1f}")
        log_session(f"Settings updated: Motion={self.motion_threshold}, White={self.white_threshold}, Black={self.black_threshold}, Clip={self.clip_limit}, Saturation={self.saturation_multiplier}")

    def update_volume_label(self, value):
        """Update music volume label."""
        percentage = int(float(value) * 100)
        self.music_volume = float(value)
        self.volume_value_label.configure(text=f"{percentage}%")
        log_session(f"Music volume set to {percentage}%")

    def select_output_dir(self):
        """Select output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = directory
            self.output_dir_label.configure(text=os.path.basename(directory) or directory)
            log_session(f"Output directory: {directory}")

    def select_music_default(self):
        """Select default music."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths["default"] = path
            self.music_label_default.configure(text=os.path.basename(path))
            log_session(f"Default music: {path}")

    def select_music_60s(self):
        """Select music for 60s videos."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[60] = path
            self.music_label_60s.configure(text=os.path.basename(path))
            log_session(f"60s music: {path}")

    def select_music_12min(self):
        """Select music for 12min videos."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[720] = path
            self.music_label_12min.configure(text=os.path.basename(path))
            log_session(f"12min music: {path}")

    def select_music_1h(self):
        """Select music for 1h videos."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[3600] = path
            self.music_label_1h.configure(text=os.path.basename(path))
            log_session(f"1h music: {path}")

    def set_schedule(self):
        """Schedule processing."""
        time_str = self.schedule_entry.get()
        try:
            schedule.every().day.at(time_str).do(self.start_processing)
            log_session(f"Scheduled at {time_str}")
            messagebox.showinfo("Success", f"Processing scheduled for {time_str}")
            threading.Thread(target=self.run_scheduler, daemon=True).start()
        except schedule.ScheduleValueError:
            messagebox.showerror("Error", "Invalid format. Use HH:MM")
            log_session(f"Invalid schedule format: {time_str}")

    def run_scheduler(self):
        """Run scheduled tasks."""
        while True:
            schedule.run_pending()
            time.sleep(60)

    def browse_files(self):
        """Browse for video files."""
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv")])
        if files:
            self.input_files.extend(files)
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if self.settings_tab:
                self.initialize_preview()
            log_session(f"Selected files: {', '.join(files)}")

    def on_drop(self, event):
        """Handle drag-and-drop."""
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if os.path.splitext(f)[1].lower() in [".mp4", ".avi", ".mkv", ".mov", ".wmv"]]
        if valid_files:
            self.input_files.extend(valid_files)
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if self.settings_tab:
                self.initialize_preview()
            log_session(f"Dropped files: {', '.join(valid_files)}")

    def initialize_preview(self):
        """Initialize video preview."""
        if self.input_files and not self.preview_cap:
            self.preview_cap = cv2.VideoCapture(self.input_files[0])
            if self.preview_cap.isOpened():
                self.fps = max(self.preview_cap.get(cv2.CAP_PROP_FPS), 1)
                self.total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.preview_slider.configure(to=self.total_frames - 1, number_of_steps=self.total_frames - 1)
                self.preview_button.configure(state="normal")
                log_session(f"Preview initialized for {self.input_files[0]}")
            else:
                log_session(f"Failed to initialize preview: {self.input_files[0]}")
                self.preview_cap = None

    def toggle_preview(self):
        """Toggle video preview."""
        if self.preview_running:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self):
        """Start preview."""
        if not self.input_files or self.preview_running:
            return
        if not self.preview_cap or not self.preview_cap.isOpened():
            self.initialize_preview()
            if not self.preview_cap or not self.preview_cap.isOpened():
                return
        self.preview_running = True
        self.preview_button.configure(text="Stop Preview")
        self.preview_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.preview_thread.start()
        log_session("Preview started")

    def stop_preview(self):
        """Stop preview."""
        self.preview_running = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            self.preview_thread = None
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        self.preview_button.configure(text="Start Preview")
        self.preview_label.configure(image=self.blank_ctk_image)
        with self.preview_queue.mutex:
            self.preview_queue.queue.clear()
        log_session("Preview stopped")

    def read_frames(self):
        """Read frames for preview."""
        frame_interval = 1 / self.fps
        while self.preview_running and self.preview_cap.isOpened():
            start_time = time.time()
            current_frame = int(self.preview_slider.get())
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.preview_cap.read()
            if not ret:
                self.preview_slider.set(0)
                continue
            frame = cv2.resize(frame, (200, 150), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
            try:
                self.preview_queue.put_nowait((ctk_img, current_frame + 1))
            except queue.Full:
                continue
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))
        log_session("Preview frame reading stopped")

    def update_preview(self):
        """Update preview display."""
        if self.preview_running:
            try:
                ctk_img, next_frame = self.preview_queue.get_nowait()
                self.preview_label.configure(image=ctk_img)
                self.preview_image = ctk_img
                self.preview_slider.set(next_frame)
            except queue.Empty:
                pass
        self.root.after(33, self.update_preview)

    def seek_preview(self, frame_idx):
        """Seek to specific frame in preview."""
        if not self.preview_running and self.preview_cap and self.preview_cap.isOpened():
            frame_idx = int(frame_idx)
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.preview_cap.read()
            if ret:
                frame = cv2.resize(frame, (200, 150), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
                self.preview_label.configure(image=ctk_img)
                self.preview_image = ctk_img

    def start_processing(self):
        """Start video processing with validations."""
        if not self.input_files:
            messagebox.showwarning("Warning", "No files selected.")
            return
        selected_videos = []
        if self.generate_60s.get():
            selected_videos.append(("60s", 59))
        if self.generate_12min.get():
            selected_videos.append(("12min", 720))
        if self.generate_1h.get():
            selected_videos.append(("1h", 3600))
        custom_duration = self.custom_duration_entry.get()
        if custom_duration:
            try:
                duration = int(custom_duration)
                if duration <= 0:
                    raise ValueError
                selected_videos.append((f"{duration}s", duration))
            except ValueError:
                messagebox.showerror("Error", "Invalid custom duration. Must be positive integer.")
                return
        if not selected_videos:
            messagebox.showwarning("Warning", "Select at least one duration.")
            return
        total_input_size = sum(os.path.getsize(f) for f in self.input_files if os.path.exists(f))
        required = total_input_size * 2
        output_path = self.output_dir or os.path.dirname(self.input_files[0] if self.input_files else '.')
        free_space = psutil.disk_usage(output_path).free
        if free_space < required:
            messagebox.showwarning("Low Disk Space", "May not have enough disk space.")
            log_session("Low disk space warning")
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        for widget in self.progress_frame.winfo_children():
            widget.destroy()
        self.progress_rows = {}
        for file in self.input_files:
            row_frame = ctk.CTkFrame(self.progress_frame)
            row_frame.pack(fill='x', pady=2)
            file_label = ctk.CTkLabel(row_frame, text=os.path.basename(file))
            file_label.pack(side=tk.LEFT, padx=5)
            status_label = ctk.CTkLabel(row_frame, text="Pending")
            status_label.pack(side=tk.LEFT, padx=5)
            progress_bar = ctk.CTkProgressBar(row_frame, width=200)
            progress_bar.pack(side=tk.LEFT, padx=5)
            progress_bar.set(0)
            cancel_button = ctk.CTkButton(row_frame, text="Cancel", command=lambda f=file: self.cancel_file(f), width=60)
            cancel_button.pack(side=tk.RIGHT, padx=5)
            self.progress_rows[file] = {"status": status_label, "progress": progress_bar, "cancel": cancel_button}
        self.switch_60s.configure(state="disabled")
        self.switch_12min.configure(state="disabled")
        self.switch_1h.configure(state="disabled")
        self.custom_duration_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.cancel_button.configure(state="normal")
        self.cancel_event.clear()
        self.pause_event.clear()
        self.paused = False
        self.start_time = time.time()
        self.analytics_data = []
        threading.Thread(target=self.process_video_thread, args=(selected_videos,)).start()
        log_session("Processing started")

    def toggle_pause(self):
        """Toggle pause/resume."""
        if self.paused:
            self.paused = False
            self.pause_event.clear()
            self.pause_button.configure(text="Pause")
            log_session("Processing resumed")
        else:
            self.paused = True
            self.pause_event.set()
            self.pause_button.configure(text="Resume")
            log_session("Processing paused")

    def cancel_processing(self):
        """Cancel all processing."""
        self.cancel_event.set()
        log_session("All processing canceled")

    def cancel_file(self, file):
        """Cancel processing for specific file."""
        self.cancel_event.set()
        if file in self.progress_rows:
            self.progress_rows[file]["status"].configure(text="Canceled")
            self.progress_rows[file]["cancel"].configure(state="disabled")
        log_session(f"Canceled file: {file}")

    def process_video_thread(self, selected_videos):
        """Process videos in thread."""
        output_format = self.output_format_var.get()
        total_tasks = len(self.input_files) * (len(selected_videos) + 1)
        task_count_queue = queue.Queue()
        task_count_queue.put(0)
        has_error = False
        with ThreadPoolExecutor(max_workers=self.worker_processes) as executor:
            futures = [executor.submit(self.process_single_video, input_file, selected_videos, output_format, total_tasks, task_count_queue)
                       for input_file in self.input_files]
            for future in futures:
                result = future.result()
                if result is None:
                    has_error = True
                    break
                else:
                    elapsed = time.time() - self.start_time
                    self.queue.put(("complete", input_file, result, elapsed))
        if not has_error and self.analytics_data:
            self.root.after(0, self.show_analytics_dashboard)
        self.root.after(0, self.reset_ui)
        log_session("Processing thread finished")

    def process_single_video(self, input_file, selected_videos, output_format, total_tasks, task_count_queue):
        """Process single video."""
        try:
            base, _ = os.path.splitext(input_file)
            output_files = {}
            task_count = task_count_queue.get()
            self.queue.put(("task_start", input_file, "Motion Detection", task_count / total_tasks * 100))
            task_count += 1
            def motion_progress_callback(progress, current, total, remaining):
                self.queue.put(("progress", input_file, "Motion Detection", progress, current, total, remaining))
            selected_indices, motion_scores = self.get_selected_indices(input_file, motion_progress_callback)
            if selected_indices is None:
                self.queue.put(("canceled", input_file, "Processing canceled"))
                return None
            if not selected_indices:
                self.queue.put(("canceled", input_file, "No frames selected"))
                return None
            for task_name, duration in selected_videos:
                if self.cancel_event.is_set():
                    self.queue.put(("canceled", input_file, "Canceled"))
                    return None
                output_file = f"{base}_{task_name}.{output_format}"
                if self.output_dir:
                    output_file = os.path.join(self.output_dir, os.path.basename(output_file))
                self.queue.put(("task_start", input_file, f"Generating {task_name}", task_count / total_tasks * 100))
                task_count += 1
                def progress_callback(progress, current, total, remaining):
                    self.queue.put(("progress", input_file, f"Generating {task_name}", progress, current, total, remaining))
                def status_callback(status):
                    self.queue.put(("status", input_file, status))
                error, frames_processed, motion_events, proc_time = self.generate_output_video(
                    input_file, output_file, duration, selected_indices, progress_callback, status_callback
                )
                if error:
                    self.queue.put(("canceled", input_file, error))
                    return None
                output_files[task_name] = output_file
                self.analytics_data.append({
                    "file": os.path.basename(input_file),
                    "duration": duration,
                    "frames_processed": frames_processed,
                    "motion_events": motion_events,
                    "processing_time": proc_time,
                    "motion_scores": motion_scores
                })
            task_count_queue.put(task_count)
            return output_files
        except Exception as e:
            self.queue.put(("canceled", input_file, str(e)))
            return None

    def get_selected_indices(self, input_path, progress_callback=None):
        """Identify frames with motion."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            log_session(f"Cannot open {input_path}")
            return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prev_frame_resized = None
        selected_indices = []
        motion_scores = []
        start_time = time.time()
        for frame_idx in range(total_frames):
            if self.cancel_event.is_set():
                cap.release()
                return None, None
            while self.pause_event.is_set() and not self.cancel_event.is_set():
                time.sleep(0.1)
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (640, 360))
            if prev_frame_resized is not None:
                score = self.compute_motion_score(prev_frame_resized, frame_resized)
                motion_scores.append(score)
                if score > self.motion_threshold and not self.is_white_or_black_frame(frame_resized):
                    selected_indices.append(frame_idx)
            prev_frame_resized = frame_resized
            if frame_idx % 100 == 0 and progress_callback:
                elapsed = time.time() - start_time
                rate = frame_idx / elapsed if elapsed > 0 else 0
                remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
                progress = (frame_idx / total_frames) * 100
                progress_callback(progress, frame_idx, total_frames, remaining)
        cap.release()
        return selected_indices, motion_scores

    def compute_motion_score(self, prev_frame, current_frame, threshold=30):
        """Compute motion score between frames."""
        if prev_frame is None or current_frame is None:
            return 0
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        score = np.sum(diff > threshold, dtype=np.uint32)
        return score

    def is_white_or_black_frame(self, frame):
        """Check if frame is overly white or black."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness > self.white_threshold or mean_brightness < self.black_threshold

    def normalize_frame(self, frame, output_resolution, clip_limit, saturation_multiplier):
        """Normalize and enhance frame."""
        try:
            frame = cv2.resize(frame, output_resolution, interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, saturation_multiplier, dtype=cv2.CV_8U)
            hsv_enhanced = cv2.merge((h, s, v))
            return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        except Exception as e:
            log_session(f"Frame normalization error: {str(e)}")
            return None

    def process_frame_batch(self, input_path, clip_limit, saturation_multiplier, rotate, temp_dir, tasks, output_resolution):
        """Process batch of frames."""
        results = []
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return results
        for frame_idx, order in tasks:
            if self.cancel_event.is_set():
                cap.release()
                return results
            while self.pause_event.is_set() and not self.cancel_event.is_set():
                time.sleep(0.1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            normalized_frame = self.normalize_frame(frame, output_resolution, clip_limit, saturation_multiplier)
            if normalized_frame is None:
                continue
            if rotate:
                normalized_frame = cv2.rotate(normalized_frame, cv2.ROTATE_90_CLOCKWISE)
            temp_path = os.path.join(temp_dir, f"frame_{order:04d}.jpg")
            cv2.imwrite(temp_path, normalized_frame)
            results.append(order)
        cap.release()
        return results

    def probe_video_resolution(self, video_path):
        """Probe video resolution using FFmpeg."""
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
        output = subprocess.check_output(cmd).decode().strip()
        width, height = map(int, output.split(','))
        return width, height

    def generate_output_video(self, input_path, output_path, desired_duration, selected_indices, progress_callback=None, status_callback=None):
        """Generate output video."""
        try:
            if status_callback:
                status_callback("Opening video...")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return "Failed to open video", 0, 0, 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames <= 0 or fps <= 0:
                return "Invalid video properties", 0, 0, 0
            width, height = self.probe_video_resolution(input_path)
            rotate = desired_duration <= 60 and width > height
            start_time = time.time()
            with tempfile.TemporaryDirectory() as temp_dir:
                target_frames_count = int(desired_duration * fps)
                if len(selected_indices) > target_frames_count:
                    step = len(selected_indices) / target_frames_count
                    final_indices = [selected_indices[int(i * step)] for i in range(target_frames_count)]
                else:
                    final_indices = selected_indices
                if status_callback:
                    status_callback("Processing frames...")
                frame_tasks = [(idx, i) for i, idx in enumerate(final_indices)]
                task_batches = [frame_tasks[i:i + self.batch_size] for i in range(0, len(frame_tasks), self.batch_size)]
                with Pool(processes=self.worker_processes) as pool:
                    partial_process = functools.partial(
                        self.process_frame_batch,
                        input_path,
                        self.clip_limit,
                        self.saturation_multiplier,
                        rotate,
                        temp_dir,
                        output_resolution=self.output_resolution
                    )
                    results = pool.map(partial_process, task_batches)
                    frame_counter = sum(len(batch) for batch in results)
                if frame_counter == 0:
                    return "No frames processed", 0, 0, 0
                num_frames = frame_counter
                new_fps = num_frames / desired_duration if num_frames < target_frames_count else fps
                if status_callback:
                    status_callback("Creating video...")
                temp_final_path = f"temp_final_{uuid.uuid4().hex}.{output_format}"
                cmd = ['ffmpeg', '-framerate', str(new_fps), '-i', os.path.join(temp_dir, 'frame_%04d.jpg'), '-s', f"{self.output_resolution[0]}x{self.output_resolution[1]}"]
                try:
                    subprocess.run(['ffmpeg', '-hwaccels'], check=True)
                    cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
                except:
                    cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
                cmd.extend(['-pix_fmt', 'yuv420p', '-r', str(new_fps)])
                if self.watermark_text:
                    cmd.extend(['-vf', f'drawtext=text={self.watermark_text}:fontcolor=white:fontsize=24:x=10:y=10'])
                if self.custom_ffmpeg_args:
                    cmd.extend(self.custom_ffmpeg_args)
                cmd.extend(['-y', temp_final_path])
                subprocess.run(cmd, check=True)
                music_path = self.music_paths.get(desired_duration, self.music_paths.get("default"))
                if music_path and os.path.exists(music_path):
                    if status_callback:
                        status_callback("Adding music...")
                    cmd = [
                        'ffmpeg', '-i', temp_final_path, '-stream_loop', '-1', '-i', music_path,
                        '-filter_complex', f"[1:a]volume={self.music_volume}[a]",
                        '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                    ]
                    subprocess.run(cmd, check=True)
                else:
                    if status_callback:
                        status_callback("Adding silent audio...")
                    cmd = [
                        'ffmpeg', '-i', temp_final_path, '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                        '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                    ]
                    subprocess.run(cmd, check=True)
                os.remove(temp_final_path)
                if progress_callback:
                    progress_callback(100, frame_counter, len(final_indices), 0)
                return None, frame_counter, len(selected_indices), time.time() - start_time
        except Exception as e:
            log_session(f"Video generation error: {str(e)}")
            return str(e), 0, 0, 0

    def process_queue(self):
        """Process UI update queue."""
        try:
            while True:
                message = self.queue.get_nowait()
                msg_type, *args = message
                if msg_type == "task_start":
                    file, task_name, progress = args
                    if file in self.progress_rows:
                        self.progress_rows[file]["status"].configure(text=task_name)
                        self.progress_rows[file]["progress"].set(progress / 100)
                elif msg_type == "progress":
                    file, task_name, progress_value, current, total, remaining = args
                    if file in self.progress_rows:
                        self.progress_rows[file]["status"].configure(text=f"{task_name} ({progress_value:.2f}%)")
                        self.progress_rows[file]["progress"].set(progress_value / 100)
                elif msg_type == "status":
                    file, status = args
                    if file in self.progress_rows:
                        self.progress_rows[file]["status"].configure(text=status)
                elif msg_type == "complete":
                    file, output_files, elapsed = args
                    if file in self.progress_rows:
                        self.progress_rows[file]["status"].configure(text="Completed")
                        self.progress_rows[file]["progress"].set(1)
                        self.progress_rows[file]["cancel"].configure(state="disabled")
                    for task, out_file in output_files.items():
                        file_frame = ctk.CTkFrame(self.output_frame)
                        file_frame.pack(fill='x', pady=2)
                        label = ctk.CTkLabel(file_frame, text=f"{task}: {out_file}")
                        label.pack(side=tk.LEFT, padx=5)
                        label.bind("<Button-1>", lambda e, f=out_file: self.open_file(f))
                        upload_button = ctk.CTkButton(file_frame, text="Upload to YouTube", command=lambda f=out_file, t=task, b=upload_button: self.start_upload(f, t, b))
                        upload_button.pack(side=tk.RIGHT, padx=5)
                elif msg_type == "canceled":
                    file, reason = args
                    if file in self.progress_rows:
                        self.progress_rows[file]["status"].configure(text=f"Canceled: {reason}")
                        self.progress_rows[file]["progress"].set(0)
                        self.progress_rows[file]["cancel"].configure(state="disabled")
                elif msg_type == "upload_progress":
                    file, progress = args
                    if file in self.progress_rows:
                        self.progress_rows[file]["status"].configure(text=f"Uploading ({progress:.2f}%)")
                        self.progress_rows[file]["progress"].set(progress / 100)
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)

    def open_file(self, file_path):
        """Open output file or folder."""
        try:
            os.startfile(file_path)
        except:
            try:
                subprocess.call(['open', file_path])
            except:
                subprocess.call(['xdg-open', file_path])

    def reset_ui(self):
        """Reset UI after processing."""
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.switch_1h.configure(state="normal")
        self.custom_duration_entry.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.pause_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        self.pause_event.clear()
        log_session("UI reset")

    def show_analytics_dashboard(self):
        """Show analytics dashboard with charts."""
        analytics_window = ctk.CTkToplevel(self.root)
        analytics_window.title("Analytics")
        analytics_window.geometry("800x600")
        tabview = ctk.CTkTabview(analytics_window)
        tabview.pack(fill="both", expand=True, pady=10, padx=10)
        for data in self.analytics_data:
            file_tab = tabview.add(data["file"])
            info_frame = ctk.CTkFrame(file_tab)
            info_frame.pack(pady=5)
            ctk.CTkLabel(info_frame, text=f"Duration: {data['duration']}s").pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(info_frame, text=f"Frames: {data['frames_processed']}").pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(info_frame, text=f"Motion Events: {data['motion_events']}").pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(info_frame, text=f"Time: {data['processing_time']:.2f}s").pack(side=tk.LEFT, padx=5)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(data["motion_scores"])
            ax.set_title("Motion Scores Over Time")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Motion Score")
            canvas = FigureCanvasTkAgg(fig, master=file_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        log_session("Analytics dashboard shown")

    def start_upload(self, file_path, task_name, button):
        """Start YouTube upload."""
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
        if not validate_video_file(file_path):
            messagebox.showerror("Error", "Invalid video file.")
            return
        if not check_network_stability():
            messagebox.showerror("Error", "Network unstable.")
            return
        button.configure(state="disabled", text="Uploading...")
        threading.Thread(target=self.upload_to_youtube, args=(file_path, task_name, button)).start()
        log_session(f"Upload started for {file_path}")

    def upload_to_youtube(self, file_path, task_name, button):
        """Upload video to YouTube."""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                youtube = self.get_youtube_client()
                if not youtube:
                    messagebox.showerror("Error", "Failed to authenticate with YouTube.")
                    break
                duration_str = task_name
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                title = file_name + (" #shorts" if "60s" in duration_str else "")
                description = "Uploaded via Bird Box Video Processor" + (" #shorts" if "60s" in duration_str else "")
                tags = ['bird', 'nature', 'video'] + (['#shorts'] if "60s" in duration_str else [])
                body = {
                    'snippet': {'title': title, 'description': description, 'tags': tags, 'categoryId': '22'},
                    'status': {'privacyStatus': 'unlisted'}
                }
                media = MediaFileUpload(file_path, resumable=True, chunksize=512 * 1024)
                request = youtube.videos().insert(part='snippet,status', body=body, media_body=media)
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        progress = status.progress() * 100
                        self.queue.put(("upload_progress", file_path, progress))
                messagebox.showinfo("Success", f"Uploaded: https://youtu.be/{response['id']}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    messagebox.showerror("Error", f"Upload failed: {str(e)}")
            finally:
                button.configure(state="normal", text="Upload to YouTube")

    def get_youtube_client(self):
        """Authenticate YouTube API client."""
        if not hasattr(self, 'youtube_client'):
            credentials = None
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    credentials = pickle.load(token)
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'client_secrets.json',
                        scopes=['https://www.googleapis.com/auth/youtube.upload']
                    )
                    credentials = flow.run_local_server(port=0)
                with open('token.pickle', 'wb') as token:
                    pickle.dump(credentials, token)
            self.youtube_client = build('youtube', 'v3', credentials=credentials)
        return self.youtube_client

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

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()