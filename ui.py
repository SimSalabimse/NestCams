import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import json
from PIL import Image
import time
from video_processing import process_single_video, generate_output_video, get_selected_indices, compute_motion_score, is_white_or_black_frame, normalize_frame, process_frame_batch, probe_video_resolution, debug_get_selected_indices, debug_normalize_frame, debug_generate_output_video
from youtube_upload import start_upload, upload_to_youtube, get_youtube_client, debug_upload_to_youtube
from utils import log_session, ToolTip, validate_video_file, check_network_stability
import threading
import queue
from multiprocessing import Event as MpEvent
import psutil
from packaging import version
import requests
import schedule
import shutil

VERSION = "10.0.0"
UPDATE_CHANNELS = ["Stable", "Beta"]

class VideoProcessorApp:
    """Main application class for Bird Box Video Processor."""
    def __init__(self, root):
        start_time = time.time()
        log_session("Starting UI initialization")
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        log_session(f"Set appearance mode: {time.time() - start_time:.2f}s")

        # Initialize instance variables
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
        log_session(f"Initialized variables: {time.time() - start_time:.2f}s")

        # Check system specs
        try:
            self.check_system_specs()
            log_session(f"Checked system specs: {time.time() - start_time:.2f}s")
        except Exception as e:
            log_session(f"System specs check failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to verify system requirements: {str(e)}")
            raise

        # Setup GUI tabs
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(pady=10, padx=10, fill="both", expand=True)
        self.main_tab = self.tabview.add("Main")
        self.settings_tab = self.tabview.add("Settings")
        self.music_tab = self.tabview.add("Music")
        self.advanced_tab = self.tabview.add("Advanced")
        self.help_tab = self.tabview.add("Help")
        self.debug_tab = self.tabview.add("Debug")
        self.setup_main_tab()
        self.setup_settings_tab()
        self.setup_music_tab()
        self.setup_advanced_tab()
        self.setup_help_tab()
        self.setup_debug_tab()
        log_session(f"Setup GUI tabs: {time.time() - start_time:.2f}s")

        # Load settings and presets after GUI setup
        self.load_settings()
        self.load_presets()
        log_session(f"Loaded settings and presets: {time.time() - start_time:.2f}s")

        # Setup drag-and-drop
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        except ImportError:
            pass
        log_session(f"Setup drag-and-drop: {time.time() - start_time:.2f}s")

        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.browse_files())
        self.root.bind('<Control-s>', lambda e: self.start_processing() if self.start_button.cget("state") == "normal" else None)
        self.root.bind('<Control-c>', lambda e: self.cancel_processing() if self.cancel_button.cget("state") == "normal" else None)
        log_session(f"Setup keyboard shortcuts: {time.time() - start_time:.2f}s")

        # Periodic tasks
        self.root.after(50, self.process_queue)
        self.root.after(33, self.update_preview)
        threading.Thread(target=self.check_for_updates, daemon=True).start()
        log_session(f"UI initialization completed: {time.time() - start_time:.2f}s")
        self.root.resizable(True, True)
        self.root.geometry("900x700")

    def setup_main_tab(self):
        start_time = time.time()
        frame = ctk.CTkFrame(self.main_tab)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        self.browse_button = ctk.CTkButton(frame, text="Browse Files", command=self.browse_files)
        self.browse_button.pack(pady=10)
        self.label = ctk.CTkLabel(frame, text="No files selected")
        self.label.pack(pady=5)
        duration_frame = ctk.CTkFrame(frame)
        duration_frame.pack(pady=10)
        self.generate_60s = ctk.CTkSwitch(duration_frame, text="Generate 60s video")
        self.generate_60s.pack(side="left", padx=5)
        self.switch_60s = self.generate_60s
        self.generate_12min = ctk.CTkSwitch(duration_frame, text="Generate 12min video")
        self.generate_12min.pack(side="left", padx=5)
        self.switch_12min = self.generate_12min
        self.generate_1h = ctk.CTkSwitch(duration_frame, text="Generate 1h video")
        self.generate_1h.pack(side="left", padx=5)
        self.switch_1h = self.generate_1h
        custom_frame = ctk.CTkFrame(frame)
        custom_frame.pack(pady=5)
        ctk.CTkLabel(custom_frame, text="Custom Duration (s):").pack(side="left")
        self.custom_duration_entry = ctk.CTkEntry(custom_frame, width=100)
        self.custom_duration_entry.pack(side="left", padx=5)
        button_frame = ctk.CTkFrame(frame)
        button_frame.pack(pady=10)
        self.start_button = ctk.CTkButton(button_frame, text="Start", command=self.start_processing, state="disabled")
        self.start_button.pack(side="left", padx=5)
        self.pause_button = ctk.CTkButton(button_frame, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(side="left", padx=5)
        self.cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(side="left", padx=5)
        self.progress_frame = ctk.CTkFrame(frame)
        self.progress_frame.pack(fill="x", pady=10)
        self.output_frame = ctk.CTkFrame(frame)
        self.output_frame.pack(fill="x", pady=10)
        log_session(f"Main tab setup: {time.time() - start_time:.2f}s")

    def setup_settings_tab(self):
        start_time = time.time()
        frame = ctk.CTkFrame(self.settings_tab)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        motion_frame = ctk.CTkFrame(frame)
        motion_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(motion_frame, text="Motion Threshold:").pack(side="left")
        self.motion_slider = ctk.CTkSlider(motion_frame, from_=1000, to=10000, number_of_steps=90, command=self.update_settings)
        self.motion_slider.set(self.motion_threshold)
        self.motion_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.motion_value_label = ctk.CTkLabel(motion_frame, text=f"Threshold: {self.motion_threshold}", width=100)
        self.motion_value_label.pack(side="left")
        white_frame = ctk.CTkFrame(frame)
        white_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(white_frame, text="White Threshold:").pack(side="left")
        self.white_slider = ctk.CTkSlider(white_frame, from_=100, to=255, number_of_steps=155, command=self.update_settings)
        self.white_slider.set(self.white_threshold)
        self.white_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.white_value_label = ctk.CTkLabel(white_frame, text=f"White: {self.white_threshold}", width=100)
        self.white_value_label.pack(side="left")
        black_frame = ctk.CTkFrame(frame)
        black_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(black_frame, text="Black Threshold:").pack(side="left")
        self.black_slider = ctk.CTkSlider(black_frame, from_=0, to=100, number_of_steps=100, command=self.update_settings)
        self.black_slider.set(self.black_threshold)
        self.black_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.black_value_label = ctk.CTkLabel(black_frame, text=f"Black: {self.black_threshold}", width=100)
        self.black_value_label.pack(side="left")
        clip_frame = ctk.CTkFrame(frame)
        clip_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(clip_frame, text="Clip Limit:").pack(side="left")
        self.clip_slider = ctk.CTkSlider(clip_frame, from_=0.1, to=5.0, number_of_steps=49, command=self.update_settings)
        self.clip_slider.set(self.clip_limit)
        self.clip_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.clip_value_label = ctk.CTkLabel(clip_frame, text=f"Clip Limit: {self.clip_limit:.1f}", width=100)
        self.clip_value_label.pack(side="left")
        sat_frame = ctk.CTkFrame(frame)
        sat_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(sat_frame, text="Saturation Multiplier:").pack(side="left")
        self.saturation_slider = ctk.CTkSlider(sat_frame, from_=0.5, to=2.0, number_of_steps=15, command=self.update_settings)
        self.saturation_slider.set(self.saturation_multiplier)
        self.saturation_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.saturation_value_label = ctk.CTkLabel(sat_frame, text=f"Saturation: {self.saturation_multiplier:.1f}", width=100)
        self.saturation_value_label.pack(side="left")
        preview_frame = ctk.CTkFrame(frame)
        preview_frame.pack(pady=10)
        self.preview_label = ctk.CTkLabel(preview_frame, image=self.blank_ctk_image, text="")
        self.preview_label.pack()
        self.preview_slider = ctk.CTkSlider(preview_frame, from_=0, to=1, command=self.seek_preview, state="disabled")
        self.preview_slider.pack(fill="x", pady=5)
        self.preview_button = ctk.CTkButton(preview_frame, text="Start Preview", command=self.toggle_preview, state="disabled")
        self.preview_button.pack()
        log_session(f"Settings tab setup: {time.time() - start_time:.2f}s")

    def setup_music_tab(self):
        start_time = time.time()
        frame = ctk.CTkFrame(self.music_tab)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        default_frame = ctk.CTkFrame(frame)
        default_frame.pack(fill="x", pady=5)
        ctk.CTkButton(default_frame, text="Select Default Music", command=self.select_music_default).pack(side="left")
        self.music_label_default = ctk.CTkLabel(default_frame, text="No file selected")
        self.music_label_default.pack(side="left", padx=10)
        s60_frame = ctk.CTkFrame(frame)
        s60_frame.pack(fill="x", pady=5)
        ctk.CTkButton(s60_frame, text="Select 60s Music", command=self.select_music_60s).pack(side="left")
        self.music_label_60s = ctk.CTkLabel(s60_frame, text="No file selected")
        self.music_label_60s.pack(side="left", padx=10)
        min12_frame = ctk.CTkFrame(frame)
        min12_frame.pack(fill="x", pady=5)
        ctk.CTkButton(min12_frame, text="Select 12min Music", command=self.select_music_12min).pack(side="left")
        self.music_label_12min = ctk.CTkLabel(min12_frame, text="No file selected")
        self.music_label_12min.pack(side="left", padx=10)
        h1_frame = ctk.CTkFrame(frame)
        h1_frame.pack(fill="x", pady=5)
        ctk.CTkButton(h1_frame, text="Select 1h Music", command=self.select_music_1h).pack(side="left")
        self.music_label_1h = ctk.CTkLabel(h1_frame, text="No file selected")
        self.music_label_1h.pack(side="left", padx=10)
        volume_frame = ctk.CTkFrame(frame)
        volume_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(volume_frame, text="Music Volume:").pack(side="left")
        self.volume_slider = ctk.CTkSlider(volume_frame, from_=0.0, to=1.0, command=self.update_volume_label)
        self.volume_slider.set(self.music_volume)
        self.volume_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.volume_value_label = ctk.CTkLabel(volume_frame, text="100%")
        self.volume_value_label.pack(side="left")
        log_session(f"Music tab setup: {time.time() - start_time:.2f}s")

    def setup_advanced_tab(self):
        start_time = time.time()
        frame = ctk.CTkFrame(self.advanced_tab)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        outdir_frame = ctk.CTkFrame(frame)
        outdir_frame.pack(fill="x", pady=5)
        ctk.CTkButton(outdir_frame, text="Select Output Directory", command=self.select_output_dir).pack(side="left")
        self.output_dir_label = ctk.CTkLabel(outdir_frame, text="Current directory")
        self.output_dir_label.pack(side="left", padx=10)
        format_frame = ctk.CTkFrame(frame)
        format_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(format_frame, text="Output Format:").pack(side="left")
        self.output_format_var = ctk.StringVar(value="mp4")
        ctk.CTkComboBox(format_frame, values=["mp4", "avi", "mkv"], variable=self.output_format_var).pack(side="left", padx=10)
        schedule_frame = ctk.CTkFrame(frame)
        schedule_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(schedule_frame, text="Schedule (HH:MM):").pack(side="left")
        self.schedule_entry = ctk.CTkEntry(schedule_frame, width=100)
        self.schedule_entry.pack(side="left", padx=5)
        ctk.CTkButton(schedule_frame, text="Set", command=self.set_schedule).pack(side="left")
        theme_frame = ctk.CTkFrame(frame)
        theme_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side="left")
        ctk.CTkOptionMenu(theme_frame, values=["Dark", "Light"], command=self.toggle_theme).pack(side="left", padx=10)
        update_frame = ctk.CTkFrame(frame)
        update_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(update_frame, text="Update Channel:").pack(side="left")
        self.update_channel_var = ctk.StringVar(value=self.update_channel)
        ctk.CTkOptionMenu(update_frame, values=UPDATE_CHANNELS, variable=self.update_channel_var, command=lambda channel: setattr(self, 'update_channel', channel)).pack(side="left", padx=10)
        log_session(f"Advanced tab setup: {time.time() - start_time:.2f}s")

    def setup_help_tab(self):
        start_time = time.time()
        frame = ctk.CTkFrame(self.help_tab)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        help_text = "Bird Box Video Processor Help\n\n" \
                    "Main Tab: Select videos and choose durations to process.\n" \
                    "Settings Tab: Adjust thresholds for motion detection and frame processing.\n" \
                    "Music Tab: Select background music for different video lengths.\n" \
                    "Advanced Tab: Set output directory, format, schedule, theme, etc.\n" \
                    "Debug Tab: Test core functionalities without processing real files.\n\n" \
                    "For more info, visit the documentation."
        ctk.CTkLabel(frame, text=help_text, justify="left").pack(anchor="w")
        log_session(f"Help tab setup: {time.time() - start_time:.2f}s")

    def setup_debug_tab(self):
        start_time = time.time()
        frame = ctk.CTkFrame(self.debug_tab)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        ctk.CTkButton(frame, text="Test Motion Detection", command=self.debug_motion_detection).pack(pady=5)
        ctk.CTkButton(frame, text="Test Frame Normalization", command=self.debug_frame_normalization).pack(pady=5)
        ctk.CTkButton(frame, text="Test Video Generation", command=self.debug_video_generation).pack(pady=5)
        ctk.CTkButton(frame, text="Test YouTube Upload", command=self.debug_youtube_upload).pack(pady=5)
        ctk.CTkButton(frame, text="Clear Results", command=self.debug_clear_results).pack(pady=5)
        self.debug_text = ctk.CTkTextbox(frame, height=200)
        self.debug_text.pack(fill="both", expand=True, pady=10)
        log_session(f"Debug tab setup: {time.time() - start_time:.2f}s")

    def debug_motion_detection(self):
        start_time = time.time()
        log_session("Starting debug motion detection test")
        selected_indices, motion_scores = debug_get_selected_indices(self)
        result = f"Motion Detection Test:\nIndices: {selected_indices[:10]}...\nScores: {motion_scores[:10]}...\nLength: {len(selected_indices)} indices"
        self.debug_text.insert("end", result + "\n\n")
        messagebox.showinfo("Debug", result)
        self.analytics_data.append({
            "file": "debug_motion_test.mp4",
            "duration": 60,
            "frames_processed": len(selected_indices),
            "motion_events": len(selected_indices),
            "processing_time": time.time() - start_time,
            "motion_scores": motion_scores
        })
        log_session(f"Debug motion detection completed: {time.time() - start_time:.2f}s")

    def debug_frame_normalization(self):
        start_time = time.time()
        log_session("Starting debug frame normalization test")
        mock_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normalized = debug_normalize_frame(self, mock_frame)
        result = f"Frame Normalization Test:\nSuccess: {normalized is not None}\nShape: {normalized.shape if normalized is not None else 'None'}"
        self.debug_text.insert("end", result + "\n\n")
        messagebox.showinfo("Debug", result)
        log_session(f"Debug frame normalization completed: {time.time() - start_time:.2f}s")

    def debug_video_generation(self):
        start_time = time.time()
        log_session("Starting debug video generation test")
        error, frames_processed, motion_events, proc_time = debug_generate_output_video(self, "debug_video.mp4", 60)
        result = f"Video Generation Test:\nError: {error or 'None'}\nFrames: {frames_processed}\nEvents: {motion_events}\nTime: {proc_time:.2f}s"
        self.debug_text.insert("end", result + "\n\n")
        messagebox.showinfo("Debug", result)
        self.analytics_data.append({
            "file": "debug_video.mp4",
            "duration": 60,
            "frames_processed": frames_processed,
            "motion_events": motion_events,
            "processing_time": proc_time,
            "motion_scores": [1000] * frames_processed
        })
        log_session(f"Debug video generation completed: {time.time() - start_time:.2f}s")

    def debug_youtube_upload(self):
        start_time = time.time()
        log_session("Starting debug YouTube upload test")
        result = debug_upload_to_youtube(self, "debug_video.mp4", "60s")
        self.debug_text.insert("end", f"YouTube Upload Test:\nResult: {result}\n\n")
        messagebox.showinfo("Debug", f"YouTube Upload Test:\nResult: {result}")
        log_session(f"Debug YouTube upload completed: {time.time() - start_time:.2f}s")

    def debug_clear_results(self):
        start_time = time.time()
        self.debug_text.delete("1.0", "end")
        log_session(f"Debug results cleared: {time.time() - start_time:.2f}s")

    def check_system_specs(self):
        start_time = time.time()
        if not shutil.which('ffmpeg'):
            log_session("FFmpeg not found")
            messagebox.showerror("Error", "FFmpeg is not installed or not in PATH. Please install FFmpeg to continue.")
            raise RuntimeError("FFmpeg not found")
        cpu_cores = psutil.cpu_count(logical=True)
        min_cores = 2
        if cpu_cores < min_cores:
            log_session(f"Low CPU cores: {cpu_cores} (recommended: {min_cores})")
            messagebox.showwarning("Warning", f"Low CPU cores detected ({cpu_cores}). Recommended: {min_cores} or more.")
        memory = psutil.virtual_memory()
        free_memory_mb = memory.available / (1024 * 1024)
        min_memory_mb = 4096
        if free_memory_mb < min_memory_mb:
            log_session(f"Low memory: {free_memory_mb:.2f}MB (recommended: {min_memory_mb}MB)")
            messagebox.showwarning("Warning", f"Low memory detected ({free_memory_mb:.2f}MB). Recommended: {min_memory_mb}MB or more.")
        default_output_dir = os.getcwd()
        disk = psutil.disk_usage(default_output_dir)
        free_space_gb = disk.free / (1024 * 1024 * 1024)
        min_space_gb = 10
        if free_space_gb < min_space_gb:
            log_session(f"Low disk space: {free_space_gb:.2f}GB (recommended: {min_space_gb}GB)")
            messagebox.showwarning("Warning", f"Low disk space detected ({free_space_gb:.2f}GB). Recommended: {min_space_gb}GB or more.")
        log_session(f"System specs: CPU cores={cpu_cores}, Free memory={free_memory_mb:.2f}MB, Free disk space={free_space_gb:.2f}GB")
        log_session(f"System specs check completed: {time.time() - start_time:.2f}s")

    def check_for_updates(self):
        start_time = time.time()
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
        log_session(f"Update check: {time.time() - start_time:.2f}s")

    def toggle_theme(self, theme):
        start_time = time.time()
        ctk.set_appearance_mode(theme.lower())
        log_session(f"Theme changed to {theme}: {time.time() - start_time:.2f}s")

    def update_settings(self, value):
        start_time = time.time()
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
        log_session(f"Settings updated: Motion={self.motion_threshold}, White={self.white_threshold}, Black={self.black_threshold}, Clip={self.clip_limit}, Saturation={self.saturation_multiplier}, Time: {time.time() - start_time:.2f}s")

    def update_volume_label(self, value):
        start_time = time.time()
        percentage = int(float(value) * 100)
        self.music_volume = float(value)
        self.volume_value_label.configure(text=f"{percentage}%")
        log_session(f"Music volume set to {percentage}%: {time.time() - start_time:.2f}s")

    def select_output_dir(self):
        start_time = time.time()
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = directory
            self.output_dir_label.configure(text=os.path.basename(directory) or directory)
            log_session(f"Output directory: {directory}, Time: {time.time() - start_time:.2f}s")

    def select_music_default(self):
        start_time = time.time()
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths["default"] = path
            self.music_label_default.configure(text=os.path.basename(path))
            log_session(f"Default music: {path}, Time: {time.time() - start_time:.2f}s")

    def select_music_60s(self):
        start_time = time.time()
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[60] = path
            self.music_label_60s.configure(text=os.path.basename(path))
            log_session(f"60s music: {path}, Time: {time.time() - start_time:.2f}s")

    def select_music_12min(self):
        start_time = time.time()
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[720] = path
            self.music_label_12min.configure(text=os.path.basename(path))
            log_session(f"12min music: {path}, Time: {time.time() - start_time:.2f}s")

    def select_music_1h(self):
        start_time = time.time()
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[3600] = path
            self.music_label_1h.configure(text=os.path.basename(path))
            log_session(f"1h music: {path}, Time: {time.time() - start_time:.2f}s")

    def set_schedule(self):
        start_time = time.time()
        time_str = self.schedule_entry.get()
        try:
            schedule.every().day.at(time_str).do(self.start_processing)
            log_session(f"Scheduled at {time_str}")
            messagebox.showinfo("Success", f"Processing scheduled for {time_str}")
            threading.Thread(target=self.run_scheduler, daemon=True).start()
        except schedule.ScheduleValueError:
            messagebox.showerror("Error", "Invalid format. Use HH:MM")
            log_session(f"Invalid schedule format: {time_str}")
        log_session(f"Schedule set: {time.time() - start_time:.2f}s")

    def run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(60)

    def browse_files(self):
        start_time = time.time()
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv")])
        if files:
            self.input_files.extend(files)
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if self.settings_tab:
                self.initialize_preview()
            log_session(f"Selected files: {', '.join(files)}, Time: {time.time() - start_time:.2f}s")

    def on_drop(self, event):
        start_time = time.time()
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if os.path.splitext(f)[1].lower() in [".mp4", ".avi", ".mkv", ".mov", ".wmv"]]
        if valid_files:
            self.input_files.extend(valid_files)
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if self.settings_tab:
                self.initialize_preview()
            log_session(f"Dropped files: {', '.join(valid_files)}, Time: {time.time() - start_time:.2f}s")

    def initialize_preview(self):
        start_time = time.time()
        import cv2
        if self.input_files and not self.preview_cap:
            self.preview_cap = cv2.VideoCapture(self.input_files[0])
            if self.preview_cap.isOpened():
                self.fps = max(self.preview_cap.get(cv2.CAP_PROP_FPS), 1)
                self.total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.preview_slider.configure(to=self.total_frames - 1, number_of_steps=self.total_frames - 1)
                self.preview_button.configure(state="normal")
                log_session(f"Preview initialized for {self.input_files[0]}, Time: {time.time() - start_time:.2f}s")
            else:
                log_session(f"Failed to initialize preview: {self.input_files[0]}, Time: {time.time() - start_time:.2f}s")
                self.preview_cap = None

    def toggle_preview(self):
        start_time = time.time()
        if self.preview_running:
            self.stop_preview()
        else:
            self.start_preview()
        log_session(f"Toggle preview: {time.time() - start_time:.2f}s")

    def start_preview(self):
        start_time = time.time()
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
        log_session(f"Preview started: {time.time() - start_time:.2f}s")

    def stop_preview(self):
        start_time = time.time()
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
        log_session(f"Preview stopped: {time.time() - start_time:.2f}s")

    def read_frames(self):
        import cv2
        start_time = time.time()
        frame_interval = 1 / self.fps
        while self.preview_running and self.preview_cap.isOpened():
            frame_start = time.time()
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
            elapsed = time.time() - frame_start
            time.sleep(max(0, frame_interval - elapsed))
        log_session(f"Preview frame reading stopped: {time.time() - start_time:.2f}s")

    def update_preview(self):
        start_time = time.time()
        if self.preview_running:
            try:
                ctk_img, next_frame = self.preview_queue.get_nowait()
                self.preview_label.configure(image=ctk_img)
                self.preview_image = ctk_img
                self.preview_slider.set(next_frame)
            except queue.Empty:
                pass
        self.root.after(33, self.update_preview)
        log_session(f"Update preview: {time.time() - start_time:.2f}s")

    def seek_preview(self, frame_idx):
        import cv2
        start_time = time.time()
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
        log_session(f"Seek preview: {time.time() - start_time:.2f}s")

    def start_processing(self):
        start_time = time.time()
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
        log_session(f"Processing started: {time.time() - start_time:.2f}s")

    def toggle_pause(self):
        start_time = time.time()
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
        log_session(f"Toggle pause: {time.time() - start_time:.2f}s")

    def cancel_processing(self):
        start_time = time.time()
        self.cancel_event.set()
        log_session(f"All processing canceled: {time.time() - start_time:.2f}s")

    def cancel_file(self, file):
        start_time = time.time()
        self.cancel_event.set()
        if file in self.progress_rows:
            self.progress_rows[file]["status"].configure(text="Canceled")
            self.progress_rows[file]["cancel"].configure(state="disabled")
        log_session(f"Canceled file: {file}, Time: {time.time() - start_time:.2f}s")

    def process_video_thread(self, selected_videos):
        from concurrent.futures import ThreadPoolExecutor
        start_time = time.time()
        output_format = self.output_format_var.get()
        total_tasks = len(self.input_files) * (len(selected_videos) + 1)
        task_count_queue = queue.Queue()
        task_count_queue.put(0)
        has_error = False
        with ThreadPoolExecutor(max_workers=self.worker_processes) as executor:
            futures = [executor.submit(process_single_video, self, input_file, selected_videos, output_format, total_tasks, task_count_queue)
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
        log_session(f"Processing thread finished: {time.time() - start_time:.2f}s")

    def process_queue(self):
        start_time = time.time()
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
                        upload_button = ctk.CTkButton(file_frame, text="Upload to YouTube", command=lambda f=out_file, t=task, b=upload_button: start_upload(self, f, t, b))
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
        log_session(f"Process queue: {time.time() - start_time:.2f}s")

    def open_file(self, file_path):
        start_time = time.time()
        try:
            import os
            os.startfile(file_path)
        except:
            try:
                import subprocess
                subprocess.call(['open', file_path])
            except:
                import subprocess
                subprocess.call(['xdg-open', file_path])
        log_session(f"Opened file {file_path}: {time.time() - start_time:.2f}s")

    def show_analytics_dashboard(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        start_time = time.time()
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
        log_session(f"Analytics dashboard shown: {time.time() - start_time:.2f}s")

    def reset_ui(self):
        start_time = time.time()
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.switch_1h.configure(state="normal")
        self.custom_duration_entry.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.pause_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        log_session(f"UI reset: {time.time() - start_time:.2f}s")

    def load_settings(self):
        start_time = time.time()
        if os.path.exists('settings.json'):
            try:
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                self.motion_slider.set(settings.get('motion_threshold', self.motion_threshold))
                self.white_slider.set(settings.get('white_threshold', self.white_threshold))
                self.black_slider.set(settings.get('black_threshold', self.black_threshold))
                self.clip_slider.set(settings.get('clip_limit', self.clip_limit))
                self.saturation_slider.set(settings.get('saturation_multiplier', self.saturation_multiplier))
                self.volume_slider.set(settings.get('music_volume', self.music_volume))
                self.update_settings(None)
                self.update_volume_label(self.music_volume)
                self.output_format_var.set(settings.get('output_format', 'mp4'))
                self.update_channel = settings.get('update_channel', 'Stable')
                self.update_channel_var.set(self.update_channel)
                self.music_paths = settings.get('music_paths', self.music_paths)
                self.music_label_default.configure(text=os.path.basename(self.music_paths['default']) if self.music_paths['default'] else "No file selected")
                self.music_label_60s.configure(text=os.path.basename(self.music_paths[60]) if self.music_paths[60] else "No file selected")
                self.music_label_12min.configure(text=os.path.basename(self.music_paths[720]) if self.music_paths[720] else "No file selected")
                self.music_label_1h.configure(text=os.path.basename(self.music_paths[3600]) if self.music_paths[3600] else "No file selected")
                self.output_dir = settings.get('output_dir', None)
                if self.output_dir:
                    self.output_dir_label.configure(text=os.path.basename(self.output_dir) or self.output_dir)
            except Exception as e:
                log_session(f"Failed to load settings: {str(e)}")
        log_session(f"Settings loaded: {time.time() - start_time:.2f}s")

    def load_presets(self):
        start_time = time.time()
        if os.path.exists('presets.json'):
            try:
                with open('presets.json', 'r') as f:
                    self.presets = json.load(f)
            except Exception as e:
                log_session(f"Failed to load presets: {str(e)}")
        log_session(f"Presets loaded: {time.time() - start_time:.2f}s")