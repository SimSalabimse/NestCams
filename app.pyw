import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
from PIL import Image, ImageTk
import threading
import queue
import uuid
import json
import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import logging
from datetime import datetime
import schedule
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing import (
    get_selected_indices,
    generate_output_video,
    probe_video_resolution,
    detect_orientation,
    validate_video_file,
    check_network_stability
)
from utils import (
    log_session,
    check_system_specs,
    check_disk_space,
    ToolTip,
    BATCH_SIZE,
    WORKER_PROCESSES,
    thread_lock,
    cancel_events,
    pause_event,
    VERSION,
    UPDATE_CHANNELS
)


class VideoProcessorApp:
    """Main application class for video processing GUI with YouTube upload."""
    def __init__(self, root):
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        log_session("Application started")
        self.root.resizable(True, True)
        self.root.geometry("900x700")

        check_system_specs()

        self.output_resolutions = {60: (1080, 1920), 720: (1920, 1080), 3600: (1920, 1080), "default": (1920, 1080)}
        self.theme_var = tk.StringVar(value="Dark")
        theme_frame = ctk.CTkFrame(root)
        theme_frame.pack(pady=5)
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(theme_frame, variable=self.theme_var, values=["Light", "Dark"], command=self.toggle_theme).pack(side=tk.LEFT)

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

        self.queue = queue.Queue()
        self.preview_queue = queue.Queue(maxsize=30)
        self.start_time = None
        self.preview_image = None
        self.blank_ctk_image = ctk.CTkImage(
            light_image=Image.new('RGB', (320, 180), (0, 0, 0)),
            dark_image=Image.new('RGB', (320, 180), (0, 0, 0)),
            size=(320, 180)
        )
        self.root.after(50, self.process_queue)
        self.root.after(20, self.update_preview)

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
        self.input_files = []
        self.presets = {}

        self.load_settings()
        self.load_presets()
        self.check_for_updates()

        self.root.bind("<Control-o>", lambda e: self.browse_files())
        self.root.bind("<Control-s>", lambda e: self.start_processing())
        self.root.bind("<Control-c>", lambda e: self.cancel_processing())

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_files)

    def setup_main_tab(self):
        """Setup the Main tab."""
        self.label = ctk.CTkLabel(self.main_tab, text="Select Input Video(s)")
        self.label.pack(pady=10)
        ToolTip(self.label, "Drag and drop videos here or browse.")

        self.generate_60s = tk.BooleanVar(value=True)
        self.switch_60s = ctk.CTkSwitch(self.main_tab, text="Generate 60s Video", variable=self.generate_60s)
        self.switch_60s.pack(pady=5)
        ToolTip(self.switch_60s, "Generate a 60-second condensed video.")

        self.generate_12min = tk.BooleanVar(value=True)
        self.switch_12min = ctk.CTkSwitch(self.main_tab, text="Generate 12min Video", variable=self.generate_12min)
        self.switch_12min.pack(pady=5)
        ToolTip(self.switch_12min, "Generate a 12-minute condensed video.")

        self.generate_1h = tk.BooleanVar(value=True)
        self.switch_1h = ctk.CTkSwitch(self.main_tab, text="Generate 1h Video", variable=self.generate_1h)
        self.switch_1h.pack(pady=5)
        ToolTip(self.switch_1h, "Generate a 1-hour condensed video.")

        custom_frame = ctk.CTkFrame(self.main_tab)
        custom_frame.pack(pady=5)
        ctk.CTkLabel(custom_frame, text="Custom Duration (seconds):").pack(side=tk.LEFT, padx=5)
        self.custom_duration_entry = ctk.CTkEntry(custom_frame, placeholder_text="e.g., 120")
        self.custom_duration_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(self.custom_duration_entry, "Enter custom video duration in seconds.")

        self.output_format_var = tk.StringVar(value="mp4")
        format_frame = ctk.CTkFrame(self.main_tab)
        format_frame.pack(pady=5)
        ctk.CTkLabel(format_frame, text="Output Format:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(format_frame, variable=self.output_format_var, values=["mp4", "avi", "mkv", "mov", "wmv"]).pack(side=tk.LEFT)
        ToolTip(format_frame, "Select the output video format.")

        self.browse_button = ctk.CTkButton(self.main_tab, text="Browse", command=self.browse_files)
        self.browse_button.pack(pady=5)
        ToolTip(self.browse_button, "Browse for video files.")

        self.start_button = ctk.CTkButton(self.main_tab, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=5)
        ToolTip(self.start_button, "Start processing selected videos.")

        self.pause_button = ctk.CTkButton(self.main_tab, text="Pause", command=self.pause_processing, state="disabled")
        self.pause_button.pack(pady=5, side=tk.LEFT, padx=20)
        ToolTip(self.pause_button, "Pause the processing.")

        self.resume_button = ctk.CTkButton(self.main_tab, text="Resume", command=self.resume_processing, state="disabled")
        self.resume_button.pack(pady=5, side=tk.LEFT, padx=20)
        ToolTip(self.resume_button, "Resume paused processing.")

        self.cancel_button = ctk.CTkButton(self.main_tab, text="Cancel All", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(pady=5)
        ToolTip(self.cancel_button, "Cancel all processing tasks.")

        self.progress_table = ttk.Treeview(self.main_tab, columns=("File", "Task", "Progress", "Status", "Actions"), show="headings")
        self.progress_table.heading("File", text="File")
        self.progress_table.heading("Task", text="Task")
        self.progress_table.heading("Progress", text="Progress")
        self.progress_table.heading("Status", text="Status")
        self.progress_table.heading("Actions", text="Actions")
        self.progress_table.pack(pady=10, fill="both", expand=True)
        ToolTip(self.progress_table, "Detailed progress per file and task.")

        self.output_label = ctk.CTkLabel(self.main_tab, text="Output Files:")
        self.output_label.pack(pady=10)
        self.output_frame = ctk.CTkScrollableFrame(self.main_tab)
        self.output_frame.pack(pady=5, fill='both', expand=True)

    def setup_settings_tab(self):
        """Setup the Settings tab."""
        ctk.CTkLabel(self.settings_tab, text="Motion Sensitivity").pack(pady=5)
        self.motion_slider = ctk.CTkSlider(self.settings_tab, from_=500, to=20000, number_of_steps=195, command=self.update_settings)
        self.motion_slider.set(self.motion_threshold)
        self.motion_slider.pack(pady=5)
        ToolTip(self.motion_slider, "Adjust motion detection sensitivity.")
        self.motion_value_label = ctk.CTkLabel(self.settings_tab, text=f"Threshold: {self.motion_threshold}")
        self.motion_value_label.pack(pady=5)

        ctk.CTkLabel(self.settings_tab, text="White Threshold").pack(pady=2)
        self.white_slider = ctk.CTkSlider(self.settings_tab, from_=100, to=255, number_of_steps=155, command=self.update_settings)
        self.white_slider.set(self.white_threshold)
        self.white_slider.pack(pady=2)
        ToolTip(self.white_slider, "Threshold for detecting white frames.")
        self.white_value_label = ctk.CTkLabel(self.settings_tab, text=f"White: {self.white_threshold}")
        self.white_value_label.pack(pady=2)

        ctk.CTkLabel(self.settings_tab, text="Black Threshold").pack(pady=2)
        self.black_slider = ctk.CTkSlider(self.settings_tab, from_=0, to=100, number_of_steps=100, command=self.update_settings)
        self.black_slider.set(self.black_threshold)
        self.black_slider.pack(pady=2)
        ToolTip(self.black_slider, "Threshold for detecting black frames.")
        self.black_value_label = ctk.CTkLabel(self.settings_tab, text=f"Black: {self.black_threshold}")
        self.black_value_label.pack(pady=2)

        ctk.CTkLabel(self.settings_tab, text="CLAHE Clip Limit").pack(pady=2)
        self.clip_slider = ctk.CTkSlider(self.settings_tab, from_=0.2, to=5.0, number_of_steps=96, command=self.update_settings)
        self.clip_slider.set(self.clip_limit)
        self.clip_slider.pack(pady=2)
        ToolTip(self.clip_slider, "Contrast limiting threshold for CLAHE.")
        self.clip_value_label = ctk.CTkLabel(self.settings_tab, text=f"Clip Limit: {self.clip_limit:.1f}")
        self.clip_value_label.pack(pady=2)

        ctk.CTkLabel(self.settings_tab, text="Saturation Multiplier").pack(pady=2)
        self.saturation_slider = ctk.CTkSlider(self.settings_tab, from_=0.5, to=2.0, number_of_steps=150, command=self.update_settings)
        self.saturation_slider.set(self.saturation_multiplier)
        self.saturation_slider.pack(pady=2)
        ToolTip(self.saturation_slider, "Multiplier for color saturation.")
        self.saturation_value_label = ctk.CTkLabel(self.settings_tab, text=f"Saturation: {self.saturation_multiplier:.1f}")
        self.saturation_value_label.pack(pady=2)

        ctk.CTkLabel(self.settings_tab, text="Resolution for 60s").pack(pady=5)
        self.res_60_var = tk.StringVar(value="1080x1920")
        ctk.CTkOptionMenu(self.settings_tab, variable=self.res_60_var, values=["320x180", "640x360", "1280x720", "1920x1080", "1080x1920"]).pack(pady=5)
        ToolTip(self.res_60_var, "Resolution for 60s videos (vertical for shorts).")

        ctk.CTkLabel(self.settings_tab, text="Resolution for 12min").pack(pady=5)
        self.res_12min_var = tk.StringVar(value="1920x1080")
        ctk.CTkOptionMenu(self.settings_tab, variable=self.res_12min_var, values=["320x180", "640x360", "1280x720", "1920x1080"]).pack(pady=5)

        ctk.CTkLabel(self.settings_tab, text="Resolution for 1h").pack(pady=5)
        self.res_1h_var = tk.StringVar(value="1920x1080")
        ctk.CTkOptionMenu(self.settings_tab, variable=self.res_1h_var, values=["320x180", "640x360", "1280x720", "1920x1080"]).pack(pady=5)

        ctk.CTkLabel(self.settings_tab, text="Default Resolution").pack(pady=5)
        self.res_default_var = tk.StringVar(value="1920x1080")
        ctk.CTkOptionMenu(self.settings_tab, variable=self.res_default_var, values=["320x180", "640x360", "1280x720", "1920x1080"]).pack(pady=5)

        ctk.CTkButton(self.settings_tab, text="Save Settings", command=self.save_settings).pack(pady=10)
        ctk.CTkButton(self.settings_tab, text="Reset to Default", command=self.reset_to_default).pack(pady=10)

        self.preview_frame = ctk.CTkFrame(self.settings_tab)
        self.preview_frame.pack(pady=10)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Preview", image=self.blank_ctk_image)
        self.preview_label.pack(pady=5)
        ToolTip(self.preview_label, "Video preview with enhancements applied.")

        control_frame = ctk.CTkFrame(self.preview_frame)
        control_frame.pack(pady=5)
        self.preview_button = ctk.CTkButton(control_frame, text="Start Preview", command=self.toggle_preview, state="disabled")
        self.preview_button.pack(side=tk.LEFT, padx=5)
        self.preview_slider = ctk.CTkSlider(control_frame, from_=0, to=0, number_of_steps=0, command=self.seek_preview)
        self.preview_slider.pack(side=tk.LEFT, padx=5)
        ToolTip(self.preview_slider, "Seek through the video preview.")

    def setup_music_tab(self):
        """Setup the Music tab."""
        music_settings_frame = ctk.CTkFrame(self.music_tab)
        music_settings_frame.pack(pady=10, fill="both", expand=True)
        ctk.CTkLabel(music_settings_frame, text="Music Settings").pack(pady=5)

        default_music_frame = ctk.CTkFrame(music_settings_frame)
        default_music_frame.pack(pady=2)
        ctk.CTkLabel(default_music_frame, text="Default Music:").pack(side=tk.LEFT)
        default_path = self.music_paths.get("default")
        self.music_label_default = ctk.CTkLabel(default_music_frame, text="No music selected" if not default_path else os.path.basename(default_path))
        self.music_label_default.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(default_music_frame, text="Select", command=self.select_music_default).pack(side=tk.LEFT)
        ToolTip(default_music_frame, "Default background music.")

        music_60s_frame = ctk.CTkFrame(music_settings_frame)
        music_60s_frame.pack(pady=2)
        ctk.CTkLabel(music_60s_frame, text="Music for 60s Video:").pack(side=tk.LEFT)
        path_60s = self.music_paths.get(60)
        self.music_label_60s = ctk.CTkLabel(music_60s_frame, text="No music selected" if not path_60s else os.path.basename(path_60s))
        self.music_label_60s.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_60s_frame, text="Select", command=self.select_music_60s).pack(side=tk.LEFT)
        ToolTip(music_60s_frame, "Music for 60s videos.")

        music_12min_frame = ctk.CTkFrame(music_settings_frame)
        music_12min_frame.pack(pady=2)
        ctk.CTkLabel(music_12min_frame, text="Music for 12min Video:").pack(side=tk.LEFT)
        path_12min = self.music_paths.get(720)
        self.music_label_12min = ctk.CTkLabel(music_12min_frame, text="No music selected" if not path_12min else os.path.basename(path_12min))
        self.music_label_12min.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_12min_frame, text="Select", command=self.select_music_12min).pack(side=tk.LEFT)
        ToolTip(music_12min_frame, "Music for 12min videos.")

        music_1h_frame = ctk.CTkFrame(music_settings_frame)
        music_1h_frame.pack(pady=2)
        ctk.CTkLabel(music_1h_frame, text="Music for 1h Video:").pack(side=tk.LEFT)
        path_1h = self.music_paths.get(3600)
        self.music_label_1h = ctk.CTkLabel(music_1h_frame, text="No music selected" if not path_1h else os.path.basename(path_1h))
        self.music_label_1h.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_1h_frame, text="Select", command=self.select_music_1h).pack(side=tk.LEFT)
        ToolTip(music_1h_frame, "Music for 1h videos.")

        volume_frame = ctk.CTkFrame(music_settings_frame)
        volume_frame.pack(pady=2)
        ctk.CTkLabel(volume_frame, text="Music Volume (0.0 - 1.0):").pack(side=tk.LEFT)
        self.music_volume_slider = ctk.CTkSlider(volume_frame, from_=0.0, to=1.0, number_of_steps=100, command=self.update_volume_label)
        self.music_volume_slider.set(self.music_volume)
        self.music_volume_slider.pack(side=tk.LEFT)
        self.volume_value_label = ctk.CTkLabel(volume_frame, text=f"{int(self.music_volume * 100)}%")
        self.volume_value_label.pack(side=tk.LEFT, padx=5)
        ToolTip(volume_frame, "Adjust music volume level.")

    def setup_advanced_tab(self):
        """Setup the Advanced tab."""
        update_channel_frame = ctk.CTkFrame(self.advanced_tab)
        update_channel_frame.pack(pady=5)
        ctk.CTkLabel(update_channel_frame, text="Update Channel:").pack(side=tk.LEFT)
        self.update_channel_var = tk.StringVar(value=self.update_channel)
        ctk.CTkOptionMenu(update_channel_frame, variable=self.update_channel_var, values=UPDATE_CHANNELS).pack(side=tk.LEFT)
        ToolTip(update_channel_frame, "Select update channel for software updates.")

        output_dir_frame = ctk.CTkFrame(self.advanced_tab)
        output_dir_frame.pack(pady=5)
        ctk.CTkLabel(output_dir_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_label = ctk.CTkLabel(output_dir_frame, text=self.output_dir or "Default")
        self.output_dir_label.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(output_dir_frame, text="Browse", command=self.select_output_dir).pack(side=tk.LEFT)
        ToolTip(output_dir_frame, "Select directory for output files.")

        ffmpeg_frame = ctk.CTkFrame(self.advanced_tab)
        ffmpeg_frame.pack(pady=5)
        ctk.CTkLabel(ffmpeg_frame, text="Custom FFmpeg Args:").pack(side=tk.LEFT)
        self.ffmpeg_entry = ctk.CTkEntry(ffmpeg_frame, placeholder_text="e.g., -vf scale=1280:720")
        if self.custom_ffmpeg_args:
            self.ffmpeg_entry.insert(0, " ".join(self.custom_ffmpeg_args))
        self.ffmpeg_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(ffmpeg_frame, "Custom FFmpeg arguments for video processing.")

        watermark_frame = ctk.CTkFrame(self.advanced_tab)
        watermark_frame.pack(pady=5)
        ctk.CTkLabel(watermark_frame, text="Watermark Text:").pack(side=tk.LEFT)
        self.watermark_entry = ctk.CTkEntry(watermark_frame, placeholder_text="Enter watermark")
        if self.watermark_text:
            self.watermark_entry.insert(0, self.watermark_text)
        self.watermark_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(watermark_frame, "Text to watermark on videos.")

        schedule_frame = ctk.CTkFrame(self.advanced_tab)
        schedule_frame.pack(pady=10)
        ctk.CTkLabel(schedule_frame, text="Schedule Processing (HH:MM):").pack(side=tk.LEFT)
        self.schedule_entry = ctk.CTkEntry(schedule_frame, placeholder_text="e.g., 14:30")
        self.schedule_entry.pack(pady=2)
        ctk.CTkButton(schedule_frame, text="Set Schedule", command=self.set_schedule).pack(pady=2)
        ToolTip(schedule_frame, "Schedule processing for a specific time.")

        preset_frame = ctk.CTkFrame(self.advanced_tab)
        preset_frame.pack(pady=10)
        ctk.CTkLabel(preset_frame, text="Preset Management").pack(pady=5)
        self.preset_combobox = ctk.CTkComboBox(preset_frame, values=list(self.presets.keys()))
        self.preset_combobox.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Load Preset", command=self.load_preset).pack(pady=5)
        self.preset_name_entry = ctk.CTkEntry(preset_frame, placeholder_text="Enter preset name")
        self.preset_name_entry.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Save Preset", command=self.save_preset).pack(pady=5)
        ToolTip(preset_frame, "Save/load preset configurations.")

    def setup_help_tab(self):
        """Setup the Help tab with in-app docs."""
        help_text = """
Welcome to Bird Box Video Processor!

- Main Tab: Select files, choose durations, start processing.
- Settings Tab: Adjust motion detection, enhancement parameters, preview video.
- Music Tab: Select background music per duration, adjust volume.
- Advanced Tab: Custom options, scheduling, presets.
- Use drag-and-drop for files.
- Keyboard shortcuts: Ctrl+O (open), Ctrl+S (start), Ctrl+C (cancel).
- Analytics: After processing, view charts in a popup.

For issues, check logs in the log directory.
"""
        textbox = ctk.CTkTextbox(self.help_tab, height=400)
        textbox.insert("0.0", help_text)
        textbox.pack(pady=10, fill="both", expand=True)

    def load_settings(self):
        """Load settings from a JSON file."""
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.motion_threshold = int(settings.get("motion_threshold", 3000))
                self.white_threshold = int(settings.get("white_threshold", 200))
                self.black_threshold = int(settings.get("black_threshold", 50))
                self.clip_limit = float(settings.get("clip_limit", 1.0))
                self.saturation_multiplier = float(settings.get("saturation_multiplier", 1.1))
                self.music_volume = float(settings.get("music_volume", 1.0))
                self.output_dir = settings.get("output_dir", None)
                self.custom_ffmpeg_args = settings.get("custom_ffmpeg_args", None)
                self.watermark_text = settings.get("watermark_text", None)
                self.update_channel = settings.get("update_channel", "Stable")
                loaded_res = settings.get("output_resolutions", {})
                for key in self.output_resolutions:
                    if str(key) in loaded_res:
                        self.output_resolutions[key] = tuple(map(int, loaded_res[str(key)].split('x')))
                loaded_music_paths = settings.get("music_paths", {})
                for key in self.music_paths:
                    if str(key) in loaded_music_paths:
                        self.music_paths[key] = loaded_music_paths[str(key)]
            log_session("Loaded settings from settings.json")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Could not load settings: {str(e)}")
            log_session(f"Could not load settings: {str(e)}")

    def save_settings(self):
        """Save current settings to a JSON file."""
        self.motion_threshold = int(self.motion_slider.get())
        self.white_threshold = int(self.white_slider.get())
        self.black_threshold = int(self.black_slider.get())
        self.clip_limit = float(self.clip_slider.get())
        self.saturation_multiplier = float(self.saturation_slider.get())
        self.music_volume = self.music_volume_slider.get()
        self.output_resolutions[60] = tuple(map(int, self.res_60_var.get().split('x')))
        self.output_resolutions[720] = tuple(map(int, self.res_12min_var.get().split('x')))
        self.output_resolutions[3600] = tuple(map(int, self.res_1h_var.get().split('x')))
        self.output_resolutions["default"] = tuple(map(int, self.res_default_var.get().split('x')))
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
            "output_resolutions": {str(k): f"{v[0]}x{v[1]}" for k, v in self.output_resolutions.items()}
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        log_session("Saved settings")

    def load_presets(self):
        """Load preset configurations from a JSON file."""
        try:
            with open("presets.json", "r") as f:
                self.presets = json.load(f)
            log_session("Loaded presets from presets.json")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.info("No presets found")
            log_session("No presets found")
            self.presets = {}

    def check_for_updates(self):
        """Check for software updates based on the selected channel."""
        try:
            channel = self.update_channel
            url = f"https://raw.githubusercontent.com/SimSalabimse/NestCams/main/{channel}_version.txt"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            latest_version_str = response.text.strip()
            if not any(char.isdigit() for char in latest_version_str):
                raise ValueError(f"Invalid version: '{latest_version_str}'")
            current_version = version.parse(VERSION)
            latest_version = version.parse(latest_version_str)
            if latest_version > current_version:
                messagebox.showinfo(
                    "Update Available",
                    f"Version {latest_version_str} is available for {channel} channel! Please restart to update."
                )
                log_session(f"Update available for {channel}: {latest_version_str}")
            else:
                log_session(f"No update available. Current: {VERSION}, Latest: {latest_version_str}")
        except Exception as e:
            logging.error(f"Update check failed: {str(e)}")
            log_session(f"Update check failed: {str(e)}")

    def toggle_theme(self, theme):
        """Switch between light and dark themes."""
        ctk.set_appearance_mode(theme.lower())
        log_session(f"Theme changed to {theme}")

    def update_settings(self, value):
        """Update settings values from sliders."""
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
        log_session(f"Updated settings: Motion={self.motion_threshold}, White={self.white_threshold}, "
                    f"Black={self.black_threshold}, Clip={self.clip_limit}, Saturation={self.saturation_multiplier}")

    def select_music_default(self):
        """Select default background music."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths["default"] = path
            self.music_label_default.configure(text=os.path.basename(path))
        else:
            self.music_paths["default"] = None
            self.music_label_default.configure(text="No music selected")
        log_session(f"Default music selected: {self.music_paths['default']}")

    def select_music_60s(self):
        """Select music for 60s videos."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[60] = path
            self.music_label_60s.configure(text=os.path.basename(path))
        else:
            self.music_paths[60] = None
            self.music_label_60s.configure(text="No music selected")
        log_session(f"60s music selected: {self.music_paths[60]}")

    def select_music_12min(self):
        """Select music for 12-minute videos."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[720] = path
            self.music_label_12min.configure(text=os.path.basename(path))
        else:
            self.music_paths[720] = None
            self.music_label_12min.configure(text="No music selected")
        log_session(f"12min music selected: {self.music_paths[720]}")

    def select_music_1h(self):
        """Select music for 1-hour videos."""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[3600] = path
            self.music_label_1h.configure(text=os.path.basename(path))
        else:
            self.music_paths[3600] = None
            self.music_label_1h.configure(text="No music selected")
        log_session(f"1h music selected: {self.music_paths[3600]}")

    def update_volume_label(self, value):
        """Update the music volume label."""
        percentage = int(float(value) * 100)
        self.music_volume = float(value)
        self.volume_value_label.configure(text=f"{percentage}%")
        log_session(f"Music volume set to {percentage}%")

    def reset_to_default(self):
        """Reset settings to default values."""
        self.motion_slider.set(3000)
        self.white_slider.set(200)
        self.black_slider.set(50)
        self.clip_slider.set(1.0)
        self.saturation_slider.set(1.1)
        self.res_60_var.set("1080x1920")
        self.res_12min_var.set("1920x1080")
        self.res_1h_var.set("1920x1080")
        self.res_default_var.set("1920x1080")
        self.music_volume_slider.set(1.0)
        self.output_dir = None
        self.output_dir_label.configure(text="Default")
        self.ffmpeg_entry.delete(0, tk.END)
        self.watermark_entry.delete(0, tk.END)
        self.update_channel_var.set("Stable")
        self.update_settings(0)
        self.update_volume_label(1.0)
        log_session("Reset settings to default values")

    def browse_files(self):
        """Browse and select input video files."""
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv")])
        if files:
            self.input_files.extend(files)
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if self.input_files:
                self.initialize_preview()
                self.auto_suggest_preset(self.input_files[0])
            log_session(f"Selected {len(files)} file(s): {', '.join(files)}")

    def drop_files(self, event):
        """Handle drag-and-drop files."""
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv'))]
        if valid_files:
            self.input_files.extend(valid_files)
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if self.input_files:
                self.initialize_preview()
                self.auto_suggest_preset(self.input_files[0])
            log_session(f"Dropped {len(valid_files)} valid file(s): {', '.join(valid_files)}")
        else:
            messagebox.showwarning("Invalid Files", "Dropped files are not supported video formats.")
            log_session("Dropped invalid files")

    def auto_suggest_preset(self, video_path):
        """Auto-suggest preset based on video type (e.g., wildlife detection)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        motion_scores = []
        prev_frame = None
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_scores.append(np.mean(diff))
            prev_frame = gray
        cap.release()
        if motion_scores:
            variance = np.var(motion_scores)
            if variance > 1000:
                suggested = "Wildlife" if "Wildlife" in self.presets else None
                if suggested:
                    response = messagebox.askyesno("Preset Suggestion", f"Detected wildlife video. Load '{suggested}' preset?")
                    if response:
                        self.preset_combobox.set(suggested)
                        self.load_preset()
                        log_session(f"Auto-suggested and loaded preset: {suggested}")

    def initialize_preview(self):
        """Initialize video preview for settings tab."""
        if self.input_files and not self.preview_cap:
            self.preview_cap = cv2.VideoCapture(self.input_files[0])
            if self.preview_cap.isOpened():
                self.fps = max(self.preview_cap.get(cv2.CAP_PROP_FPS), 1)
                self.total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.preview_slider.configure(to=self.total_frames - 1, number_of_steps=self.total_frames - 1)
                duration = self.total_frames / self.fps
                ctk.CTkLabel(self.preview_frame, text=f"Video duration: {duration:.2f} seconds").pack(pady=5)
                self.preview_button.configure(state="normal")
                log_session(f"Initialized preview for {self.input_files[0]}")
            else:
                logging.error(f"Failed to open video for preview: {self.input_files[0]}")
                log_session(f"Error: Failed to initialize preview for {self.input_files[0]}")
                self.preview_cap = None

    def toggle_preview(self):
        """Toggle video preview on or off."""
        if self.preview_running:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self):
        """Start the video preview."""
        if not self.input_files or self.preview_running:
            log_session("Cannot start preview: no input files or already running")
            return
        if not self.preview_cap or not self.preview_cap.isOpened():
            self.initialize_preview()
            if not self.preview_cap or not self.preview_cap.isOpened():
                log_session("Cannot start preview: failed to initialize video capture")
                return
        self.preview_running = True
        self.preview_button.configure(text="Stop Preview")
        self.preview_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.preview_thread.start()
        log_session("Preview started")

    def stop_preview(self):
        """Stop the video preview."""
        if not self.preview_running:
            log_session("Preview not running")
            return
        self.preview_running = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            if self.preview_thread.is_alive():
                logging.warning("Preview thread did not stop within timeout")
                log_session("Warning: Preview thread did not stop within timeout")
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
        """Read frames for preview in a separate thread."""
        log_session("Started frame reading thread")
        frame_interval = 1 / self.fps
        while self.preview_running and self.preview_cap.isOpened():
            start_time = time.time()
            current_frame = int(self.preview_slider.get())
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.preview_cap.read()
            if not ret:
                log_session("Reached end of video or error, looping back")
                self.preview_slider.set(0)
                continue
            enhanced_frame = normalize_frame(frame, (320, 180), self.clip_limit, self.saturation_multiplier)
            if enhanced_frame is None:
                continue
            frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 180))
            try:
                self.preview_queue.put_nowait((ctk_img, current_frame + 1))
            except queue.Full:
                log_session(f"Queue full at frame {current_frame}, dropping frame")
                continue
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)
        log_session("Frame reading thread stopped")

    def update_preview(self):
        """Update the preview display."""
        if self.preview_running:
            try:
                ctk_img, next_frame = self.preview_queue.get_nowait()
                self.preview_label.configure(image=ctk_img)
                self.preview_image = ctk_img
                self.preview_slider.set(next_frame)
            except queue.Empty:
                pass
        self.root.after(20, self.update_preview)

    def seek_preview(self, frame_idx):
        """Seek to a specific frame in the preview."""
        if self.preview_cap and self.preview_cap.isOpened():
            frame_idx = int(frame_idx)
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.preview_cap.read()
            if ret:
                enhanced_frame = normalize_frame(frame, (320, 180), self.clip_limit, self.saturation_multiplier)
                if enhanced_frame is not None:
                    frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 180))
                    self.preview_label.configure(image=ctk_img)
                    self.preview_image = ctk_img
            else:
                log_session(f"Failed to seek to frame {frame_idx}")

    def select_output_dir(self):
        """Select an output directory for processed videos."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = directory
            self.output_dir_label.configure(text=os.path.basename(directory) or directory)
            log_session(f"Selected output directory: {directory}")

    def set_schedule(self):
        """Schedule video processing for a specific time."""
        time_str = self.schedule_entry.get()
        try:
            schedule.every().day.at(time_str).do(self.start_processing)
            log_session(f"Scheduled processing at {time_str}")
            messagebox.showinfo("Success", f"Processing scheduled for {time_str}")
            threading.Thread(target=self.run_scheduler, daemon=True).start()
        except schedule.ScheduleValueError:
            messagebox.showerror("Error", "Invalid time format. Use HH:MM (e.g., 14:30)")
            log_session(f"Failed to schedule processing: Invalid time format {time_str}")

    def run_scheduler(self):
        """Run the scheduled tasks."""
        while True:
            schedule.run_pending()
            time.sleep(60)

    def start_processing(self):
        """Start the video processing workflow."""
        global cancel_events, pause_event
        if not check_disk_space():
            messagebox.showerror("Error", "Insufficient disk space. Free up at least 500 MB.")
            return
        log_session("Starting video processing")
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
                    raise ValueError("Duration must be positive")
                selected_videos.append((f"{duration}s", duration))
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                log_session(f"Invalid custom duration: {str(e)}")
                return
        if not selected_videos:
            messagebox.showwarning("Warning", "Select at least one video duration.")
            return
        self.progress_table.delete(*self.progress_table.get_children())
        self.switch_60s.configure(state="disabled")
        self.switch_12min.configure(state="disabled")
        self.switch_1h.configure(state="disabled")
        self.custom_duration_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.resume_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        pause_event.set()
        cancel_events.clear()
        self.start_time = time.time()
        self.analytics_data = []
        threading.Thread(target=self.process_video_thread, args=(selected_videos,)).start()
        log_session("Started processing thread")

    def process_video_thread(self, selected_videos):
        """Thread to process all selected videos."""
        global cancel_events, pause_event
        try:
            log_session("Processing thread started")
            output_format = self.output_format_var.get()
            total_tasks = len(self.input_files) * (len(selected_videos) + 1)
            task_count_queue = queue.Queue()
            task_count_queue.put(0)
            has_error = False

            for input_file in self.input_files:
                file_id = uuid.uuid4().hex
                base, _ = os.path.splitext(input_file)
                output_files = {}
                task_count = task_count_queue.get()
                task_id = f"{file_id}_motion"
                cancel_events[task_id] = threading.Event()
                self.queue.put(("task_start", os.path.basename(input_file), "Motion Detection", task_count / total_tasks * 100, file_id, task_id))

                def motion_progress_callback(progress, current, total, remaining):
                    self.queue.put(("progress", progress, current, total, remaining, file_id, task_id))

                selected_indices = get_selected_indices(
                    input_file, self.motion_threshold, self.white_threshold, self.black_threshold,
                    progress_callback=motion_progress_callback, task_id=task_id
                )
                if selected_indices is None:
                    self.queue.put(("canceled", "Processing canceled by user", file_id, task_id))
                    has_error = True
                    break
                if not selected_indices:
                    log_session(f"No frames selected for {input_file}")
                    self.queue.put(("canceled", "No frames selected after motion detection", file_id, task_id))
                    has_error = True
                    break

                for task_name, duration in selected_videos:
                    pause_event.wait()
                    task_id = f"{file_id}_{duration}"
                    cancel_events[task_id] = threading.Event()
                    output_file = f"{base}_{task_name}.{output_format}"
                    if self.output_dir:
                        output_file = os.path.join(self.output_dir, os.path.basename(output_file))
                    self.queue.put(("task_start", os.path.basename(input_file), task_name, task_count / total_tasks * 100, file_id, task_id))
                    task_count += 1

                    def progress_callback(progress, current, total, remaining):
                        with thread_lock:
                            self.queue.put(("progress", progress, current, total, remaining, file_id, task_id))

                    def status_callback(status):
                        self.queue.put(("status", status, file_id, task_id))

                    resolution = self.output_resolutions.get(duration, self.output_resolutions["default"])
                    error, frames_processed, motion_events, proc_time = generate_output_video(
                        input_file, output_file, duration, selected_indices, resolution,
                        clip_limit=self.clip_limit,
                        saturation_multiplier=self.saturation_multiplier,
                        output_format=output_format,
                        progress_callback=progress_callback,
                        music_paths=self.music_paths,
                        music_volume=self.music_volume,
                        status_callback=status_callback,
                        custom_ffmpeg_args=self.custom_ffmpeg_args,
                        watermark_text=self.watermark_text,
                        task_id=task_id
                    )
                    if error:
                        self.queue.put(("canceled", error, file_id, task_id))
                        has_error = True
                        break
                    else:
                        output_files[task_name] = output_file
                        self.analytics_data.append({
                            "file": os.path.basename(input_file),
                            "duration": duration,
                            "frames_processed": frames_processed,
                            "motion_events": motion_events,
                            "processing_time": proc_time
                        })
                if has_error:
                    break
                task_count_queue.put(task_count)
                elapsed = time.time() - self.start_time
                self.queue.put(("complete", output_files, elapsed, file_id))
            if not has_error and self.analytics_data:
                self.root.after(0, self.show_analytics_dashboard)
            log_session("Processing thread finished")
        except Exception as e:
            log_session(f"Error in processing thread: {str(e)}")
            self.queue.put(("canceled", str(e), None, None))

    def pause_processing(self):
        """Pause ongoing processing."""
        global pause_event
        pause_event.clear()
        self.pause_button.configure(state="disabled")
        self.resume_button.configure(state="normal")
        log_session("Processing paused")

    def resume_processing(self):
        """Resume paused processing."""
        global pause_event
        pause_event.set()
        self.pause_button.configure(state="normal")
        self.resume_button.configure(state="disabled")
        log_session("Processing resumed")

    def cancel_processing(self):
        """Cancel all ongoing processing."""
        global cancel_events
        for event in cancel_events.values():
            event.set()
        log_session("All processing canceled")

    def cancel_task(self, task_id):
        """Cancel a specific task."""
        global cancel_events
        if task_id in cancel_events:
            cancel_events[task_id].set()
            log_session(f"Canceled task {task_id}")

    def process_queue(self):
        """Process messages from the queue for UI updates."""
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)

    def handle_message(self, message):
        """Handle different types of queue messages."""
        msg_type, *args = message
        if msg_type == "task_start":
            file_name, task_name, progress, file_id, task_id = args
            self.progress_table.insert("", "end", iid=task_id, values=(file_name, task_name, "0%", "Starting", "Cancel"))
            self.progress_table.bind("<Button-1>", self.on_table_click)
            log_session(f"Task started: {file_name} - {task_name}")
        elif msg_type == "progress":
            progress_value, current, total, remaining, file_id, task_id = args
            progress_value = min(max(progress_value, 0), 100)
            self.progress_table.set(task_id, "Progress", f"{progress_value:.2f}%")
            log_session(f"Progress for {task_id}: {progress_value:.2f}%")
        elif msg_type == "status":
            status_text, file_id, task_id = args
            self.progress_table.set(task_id, "Status", status_text)
            log_session(f"Status for {task_id}: {status_text}")
        elif msg_type == "upload_progress":
            progress_value = args[0]
        elif msg_type == "complete":
            output_files, elapsed, file_id = args
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes} min {seconds} sec"
            for widget in self.output_frame.winfo_children():
                widget.destroy()
            for task, file in output_files.items():
                file_frame = ctk.CTkFrame(self.output_frame)
                file_frame.pack(fill='x', pady=2)
                label = ctk.CTkLabel(file_frame, text=f"{task}: {file}")
                label.pack(side=tk.LEFT, padx=5)
                label.bind("<Button-1>", lambda e, f=file: self.open_file(f))
                upload_button = ctk.CTkButton(file_frame, text="Upload to YouTube")
                upload_button.configure(command=lambda f=file, t=task, b=upload_button: self.start_upload(f, t, b))
                upload_button.pack(side=tk.RIGHT, padx=5)
            self.reset_ui()
            log_session(f"Processing completed in {time_str}")
        elif msg_type == "canceled":
            reason, file_id, task_id = args
            if task_id:
                self.progress_table.set(task_id, "Status", f"Canceled: {reason}")
            self.reset_ui()
            log_session(f"Processing canceled: {reason}")

    def on_table_click(self, event):
        """Handle clicks on progress table for cancel or other actions."""
        item = self.progress_table.identify_row(event.y)
        column = self.progress_table.identify_column(event.x)
        if column == "#5":
            self.cancel_task(item)

    def open_file(self, file_path):
        """Open file or folder on click."""
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            folder = os.path.dirname(file_path)
            if os.path.exists(folder):
                os.startfile(folder)
            else:
                messagebox.showerror("Error", f"File or folder not found: {file_path}")
                log_session(f"Failed to open file/folder: {file_path}")

    def reset_ui(self):
        """Reset UI elements to their initial state."""
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.switch_1h.configure(state="normal")
        self.custom_duration_entry.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.pause_button.configure(state="disabled")
        self.resume_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        log_session("UI reset to initial state")

    def get_youtube_client(self):
        """Authenticate and return a YouTube API client."""
        if not hasattr(self, 'youtube_client'):
            log_session("Initializing YouTube client")
            credentials = None
            if os.path.exists('token.pickle'):
                try:
                    with open('token.pickle', 'rb') as token:
                        credentials = pickle.load(token)
                    log_session("Loaded YouTube credentials from token.pickle")
                except (pickle.PickleError, EOFError) as e:
                    log_session(f"Error: Failed to load token.pickle: {str(e)}")
                    os.remove('token.pickle')
                    credentials = None
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    try:
                        credentials.refresh(Request())
                        log_session("Refreshed YouTube credentials")
                    except Exception as e:
                        log_session(f"Error: Failed to refresh credentials: {str(e)}")
                        credentials = None
                if not credentials:
                    if not os.path.exists('client_secrets.json'):
                        messagebox.showerror("Error", "client_secrets.json not found.")
                        log_session("Error: client_secrets.json not found")
                        return None
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'client_secrets.json',
                            scopes=['https://www.googleapis.com/auth/youtube.upload']
                        )
                        credentials = flow.run_local_server(port=0)
                        log_session("Authenticated with YouTube API")
                    except Exception as e:
                        messagebox.showerror("Error", f"Authentication failed: {str(e)}")
                        log_session(f"Error: Failed to authenticate with YouTube API: {str(e)}")
                        return None
                with open('token.pickle', 'wb') as token:
                    pickle.dump(credentials, token)
                    log_session("Saved YouTube credentials to token.pickle")
            self.youtube_client = build('youtube', 'v3', credentials=credentials)
            log_session("YouTube client initialized")
        return self.youtube_client

    def start_upload(self, file_path, task_name, button):
        """Initiate the YouTube upload process."""
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Video file not found: {file_path}")
            button.configure(state="normal", text="Upload to YouTube")
            log_session(f"Error: Video file not found for upload: {file_path}")
            return
        if not validate_video_file(file_path):
            messagebox.showerror("Error", "Invalid or corrupted video file.")
            button.configure(state="normal", text="Upload to YouTube")
            log_session(f"Error: Invalid or corrupted video file: {file_path}")
            return
        if not check_network_stability():
            messagebox.showerror("Error", "Network unstable.")
            button.configure(state="normal", text="Upload to YouTube")
            log_session("Error: Network unstable, cannot upload to YouTube")
            return
        button.configure(state="disabled", text="Uploading...")
        thread = threading.Thread(target=self.upload_to_youtube, args=(file_path, task_name, button))
        thread.start()
        log_session(f"Started YouTube upload for {file_path}")

    def upload_to_youtube(self, file_path, task_name, button):
        """Upload a video to YouTube with progress updates."""
        log_session(f"Starting upload for {file_path}")
        max_retries = 10
        for attempt in range(max_retries):
            try:
                youtube = self.get_youtube_client()
                if not youtube:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to authenticate with YouTube"))
                    return
                duration_str = task_name.split()[0]
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                title = file_name + (" #shorts" if duration_str == "60s" else "")
                description = "Uploaded via Bird Box Video Processor" + (" #shorts" if duration_str == "60s" else "")
                tags = ['bird', 'nature', 'video'] + (['#shorts'] if duration_str == "60s" else [])
                request_body = {
                    'snippet': {'title': title, 'description': description, 'tags': tags, 'categoryId': '22'},
                    'status': {'privacyStatus': 'unlisted'}
                }
                media = MediaFileUpload(file_path, resumable=True, chunksize=512 * 1024)
                request = youtube.videos().insert(part='snippet,status', body=request_body, media_body=media)
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        progress = status.progress() * 100
                        self.queue.put(("upload_progress", progress))
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Video uploaded: https://youtu.be/{response['id']}"))
                log_session(f"Upload successful: {file_path}, YouTube URL: https://youtu.be/{response['id']}")
                break
            except Exception as e:
                log_session(f"Upload failed on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Upload failed after {max_retries} attempts: {str(e)}"))
                else:
                    time.sleep(5)
            finally:
                self.root.after(0, lambda b=button: b.configure(state="normal", text="Upload to YouTube"))

    def load_preset(self):
        """Load a preset configuration."""
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

    def save_preset(self):
        """Save current settings as a preset."""
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

    def show_analytics_dashboard(self):
        """Display analytics after processing with Matplotlib charts."""
        analytics_window = ctk.CTkToplevel(self.root)
        analytics_window.title("Processing Analytics")
        analytics_window.geometry("800x600")
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

        fig, ax = plt.subplots()
        durations = [d['duration'] for d in self.analytics_data]
        motions = [d['motion_events'] for d in self.analytics_data]
        ax.bar(durations, motions, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Motion Events")
        ax.set_title("Motion Events per Video Duration")
        canvas = FigureCanvasTkAgg(fig, master=analytics_window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        log_session("Displayed analytics dashboard with charts")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()