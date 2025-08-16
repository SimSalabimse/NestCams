import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel, Label, ttk
from tkinterdnd2 import DND_FILES, Tk
from processing import (
    get_selected_indices,
    generate_output_video,
    process_frame_batch,
    probe_video_resolution,
    detect_orientation,
    normalize_frame,
    compute_motion_score,
    is_white_or_black_frame,
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
import cv2
import numpy as np
import os
import time
import threading
import queue
import uuid
import json
from PIL import Image, ImageTk
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
from multiprocessing import Pool, Event
from concurrent.futures import ThreadPoolExecutor
from packaging import version
import functools
import psutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network stability checks will be limited.")

class VideoProcessorApp:
    """Main application class for video processing GUI with YouTube upload. Improved with tabs, tooltips, etc."""
    def __init__(self, root):
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        log_session("Application started")
        self.root.resizable(True, True)
        self.root.geometry("900x700")  # Slightly larger for tabs

        check_system_specs()

        self.output_resolutions = {60: (1080, 1920), 720: (1920, 1080), 3600: (1920, 1080), "default": (1920, 1080)}  # Per-duration resolutions
        self.theme_var = tk.StringVar(value="Dark")
        theme_frame = ctk.CTkFrame(root)
        theme_frame.pack(pady=5)
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(theme_frame, variable=self.theme_var, values=["Light", "Dark"], command=self.toggle_theme).pack(side=tk.LEFT)

        # Tabview for organization
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
        self.preview_queue = queue.Queue(maxsize=30)  # Increased for smoother playback
        self.start_time = None
        self.preview_image = None
        self.blank_ctk_image = ctk.CTkImage(
            light_image=Image.new('RGB', (320, 180), (0, 0, 0)),
            dark_image=Image.new('RGB', (320, 180), (0, 0, 0)),
            size=(320, 180)  # Larger preview
        )
        self.root.after(50, self.process_queue)
        self.root.after(20, self.update_preview)  # Faster update for smoothness

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
        self.load_settings()
        self.load_presets()
        self.check_for_updates()

        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.browse_files())
        self.root.bind("<Control-s>", lambda e: self.start_processing())
        self.root.bind("<Control-c>", lambda e: self.cancel_processing())

        # Drag-and-drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_files)

        self.input_files = []

    # All the setup_tab methods, load_settings, save_settings, etc., remain the same as in the provided code.
    # Omitted for brevity, but they are unchanged except for any minor adjustments if needed.

    # The methods like initialize_preview, toggle_preview, start_preview, stop_preview, read_frames, update_preview, seek_preview remain the same.

    # Methods like select_music_*, update_volume_label, reset_to_default, browse_files, drop_files, auto_suggest_preset, select_output_dir, set_schedule, run_scheduler, start_processing, process_video_thread, pause_processing, resume_processing, cancel_processing, cancel_task, process_queue, handle_message, on_table_click, open_file, reset_ui, get_youtube_client, start_upload, upload_to_youtube, load_preset, save_preset, show_analytics_dashboard remain the same.

if __name__ == "__main__":
    root = Tk()
    app = VideoProcessorApp(root)
    root.mainloop()