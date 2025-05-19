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

try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network stability checks will be limited.")

# Version number
VERSION = "7.1.2"  # Updated to reflect optimizations and progress display changes

# Set up debug logging
logging.basicConfig(filename='upload_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up session logging
session_log_file = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
session_logger = logging.getLogger('session')
session_handler = logging.FileHandler(session_log_file)
session_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
session_logger.addHandler(session_handler)
session_logger.setLevel(logging.INFO)

# Thread lock for shared resources
thread_lock = threading.Lock()

### Helper Functions

def log_session(message):
    """Log a user-friendly message to the session log."""
    session_logger.info(message)

def compute_motion_score(prev_frame, current_frame, threshold=30):
    """Compute motion score between two frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.sum(diff > threshold, dtype=np.uint32)
    logging.debug(f"Motion score: {score}")
    return score

def is_white_or_black_frame(frame, white_threshold=200, black_threshold=50):
    """Check if frame is predominantly white or black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > white_threshold or mean_brightness < black_threshold

def normalize_frame(frame, clip_limit=1.0, saturation_multiplier=1.1):
    """Normalize frame brightness and enhance saturation."""
    try:
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
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
    except MemoryError:
        logging.error("MemoryError in normalize_frame")
        log_session("Error: Memory issue while normalizing frame")
        return None

def validate_video_file(file_path):
    """Validate video file using FFmpeg."""
    try:
        cmd = ['ffmpeg', '-i', file_path, '-f', 'null', '-']
        subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        logging.info(f"Validated video file: {file_path}")
        log_session(f"Validated video file: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Validation failed for {file_path}: {e.stderr.decode()}")
        log_session(f"Validation failed for {file_path}: {e.stderr.decode()}")
        return False

def check_network_stability():
    """Check network stability with speed test or fallback."""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code != 200:
            logging.warning(f"Network ping failed: Status {response.status_code}")
            log_session(f"Network ping failed: Status {response.status_code}")
            return False
        
        if speedtest is None:
            logging.info("No speedtest module, using ping only")
            log_session("No speedtest module, using ping only")
            return True
        
        st = speedtest.Speedtest()
        st.get_best_server()
        upload_speed = st.upload() / 1_000_000  # Mbps
        if upload_speed < 1.0:
            logging.warning(f"Upload speed too low: {upload_speed:.2f} Mbps")
            log_session(f"Upload speed too low: {upload_speed:.2f} Mbps")
            return False
        logging.info(f"Network stable: Upload speed {upload_speed:.2f} Mbps")
        log_session(f"Network stable: Upload speed {upload_speed:.2f} Mbps")
        return True
    except Exception as e:
        logging.warning(f"Network check failed: {str(e)}")
        log_session(f"Network check failed: {str(e)}")
        return False

def process_video_multi_pass(input_path, output_path, desired_duration, motion_threshold=3000, sample_interval=5, 
                             white_threshold=200, black_threshold=50, clip_limit=1.0, saturation_multiplier=1.1, 
                             output_format='mp4', progress_callback=None, cancel_event=None, music_path=None, status_callback=None):
    """Process video by saving frames as images and assembling with FFmpeg."""
    try:
        logging.info(f"Starting video processing for {input_path}")
        log_session(f"Starting video processing for {input_path}")
        if status_callback:
            status_callback("Opening video file...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {input_path}")
            log_session(f"Error: Cannot open video file: {input_path}")
            return "Failed to open video file"

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames <= 0 or fps <= 0:
            cap.release()
            logging.error(f"Invalid video properties: total_frames={total_frames}, fps={fps}")
            log_session(f"Error: Invalid video properties: total_frames={total_frames}, fps={fps}")
            return "Invalid video properties"
        logging.info(f"Video properties: {total_frames} frames, {fps} fps, {frame_width}x{frame_height}")
        log_session(f"Video properties: {total_frames} frames, {fps} fps, {frame_width}x{frame_height}")
        cap.release()

        rotate = desired_duration == 60
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Single pass for motion detection and filtering
            if status_callback:
                status_callback("Analyzing motion and filtering frames...")
            logging.info("Starting motion detection and filtering pass")
            log_session("Starting motion detection and filtering pass")
            cap = cv2.VideoCapture(input_path)
            prev_frame = None
            selected_indices = []
            for frame_idx in range(total_frames):
                if cancel_event and cancel_event.is_set():
                    cap.release()
                    return "Processing canceled by user"
                ret, frame = cap.read()
                if not ret:
                    break
                if prev_frame is not None:
                    motion_score = compute_motion_score(prev_frame, frame)
                    if motion_score > motion_threshold and not is_white_or_black_frame(frame, white_threshold, black_threshold):
                        selected_indices.append(frame_idx)
                prev_frame = frame
                if progress_callback and frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 50  # 0 to 50%
                    elapsed = time.time() - start_time
                    rate = frame_idx / elapsed if elapsed > 0 else 1e-6
                    remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
                    progress_callback(progress, frame_idx, total_frames, remaining)
            cap.release()
            logging.info(f"Motion detection and filtering completed, {len(selected_indices)} frames selected")
            log_session(f"Motion detection and filtering completed, {len(selected_indices)} frames selected")
            if status_callback:
                status_callback("Motion detection and filtering completed")

            if len(selected_indices) == 0:
                logging.warning("No frames selected for the final video")
                log_session("Warning: No frames selected for the final video")
                return "No frames written after processing"

            # Select frames for final video
            target_frames_count = int(desired_duration * fps)
            if len(selected_indices) > target_frames_count:
                step = len(selected_indices) / target_frames_count
                final_indices = [selected_indices[int(i * step)] for i in range(target_frames_count)]
            else:
                final_indices = selected_indices

            # Save selected frames as images
            if status_callback:
                status_callback("Saving processed frames...")
            logging.info("Saving processed frames")
            log_session("Saving processed frames")
            final_indices_set = set(final_indices)
            cap = cv2.VideoCapture(input_path)
            frame_counter = 0
            for frame_idx in range(total_frames):
                if frame_idx in final_indices_set:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    normalized_frame = normalize_frame(frame, clip_limit, saturation_multiplier)
                    if normalized_frame is None:
                        continue
                    if rotate:
                        normalized_frame = cv2.rotate(normalized_frame, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{frame_counter:04d}.jpg"), normalized_frame)
                    frame_counter += 1
                    if progress_callback and frame_counter % 10 == 0:
                        progress = 50 + (frame_counter / len(final_indices)) * 40  # 50 to 90%
                        elapsed = time.time() - start_time
                        rate = frame_counter / elapsed if elapsed > 0 else 1e-6
                        remaining = (len(final_indices) - frame_counter) / rate if rate > 0 else 0
                        progress_callback(progress, frame_counter, len(final_indices), remaining)
                else:
                    cap.grab()  # Skip to next frame without decoding
            cap.release()
            logging.info(f"Saved {frame_counter} frames to temporary directory")
            log_session(f"Saved {frame_counter} frames to temporary directory")
            if status_callback:
                status_callback("Frames saved")

            # Calculate the number of frames
            num_frames = frame_counter

            # Calculate the required frame rate to achieve desired_duration
            if num_frames < target_frames_count:
                new_fps = num_frames / desired_duration
                logging.info(f"Adjusting frame rate to {new_fps:.2f} fps to meet desired duration")
                log_session(f"Adjusting frame rate to {new_fps:.2f} fps to meet desired duration")
            else:
                new_fps = fps  # Use original fps if we have enough frames

            # Create video with FFmpeg using new_fps
            if status_callback:
                status_callback("Creating video from frames...")
            logging.info("Creating video from frames with FFmpeg")
            log_session("Creating video from frames with FFmpeg")
            if progress_callback:
                progress_callback(90, 0, 0, 0)  # Set to 90% before FFmpeg
            temp_final_path = f"temp_final_{uuid.uuid4().hex}.mp4"
            cmd = [
                'ffmpeg', '-framerate', str(new_fps), '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', temp_final_path
            ]
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            logging.info("Video created from frames with adjusted frame rate")
            log_session("Video created from frames with adjusted frame rate")

            # Add music or silent audio
            if music_path and os.path.exists(music_path):
                if status_callback:
                    status_callback("Adding music...")
                logging.info("Adding music with FFmpeg")
                log_session("Adding music with FFmpeg")
                cmd = [
                    'ffmpeg', '-i', temp_final_path, '-i', music_path,
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                ]
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                logging.info(f"Music added to {output_path}")
                log_session(f"Music added to {output_path}")
            else:
                if status_callback:
                    status_callback("Adding silent audio...")
                logging.info("Adding silent audio with FFmpeg")
                log_session("Adding silent audio with FFmpeg")
                cmd = [
                    'ffmpeg', '-i', temp_final_path, '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                ]
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                logging.info(f"Silent audio added to {output_path}")
                log_session(f"Silent audio added to {output_path}")

            os.remove(temp_final_path)
            if progress_callback:
                progress_callback(100, 0, 0, 0)  # Set to 100% after completion
            logging.info(f"Processing completed successfully for {output_path}")
            log_session(f"Processing completed successfully for {output_path}")
            return None

    except Exception as e:
        logging.error(f"Unexpected error in video processing: {str(e)}", exc_info=True)
        log_session(f"Error: Unexpected error in video processing: {str(e)}")
        return f"Error: {str(e)}"

### Main Application Class

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize GUI with video processing."""
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        log_session("Application started")

        self.theme_var = tk.StringVar(value="Dark")
        theme_frame = ctk.CTkFrame(root)
        theme_frame.pack(pady=5)
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(theme_frame, variable=self.theme_var, values=["Light", "Dark"], command=self.toggle_theme).pack(side=tk.LEFT)
        
        self.label = ctk.CTkLabel(root, text="Select Input Video(s)")
        self.label.pack(pady=10)
        
        self.generate_60s = tk.BooleanVar(value=True)
        self.switch_60s = ctk.CTkSwitch(root, text="Generate 60s Video", variable=self.generate_60s)
        self.switch_60s.pack(pady=5)
        
        self.generate_12min = tk.BooleanVar(value=True)
        self.switch_12min = ctk.CTkSwitch(root, text="Generate 12min Video", variable=self.generate_12min)
        self.switch_12min.pack(pady=5)
        
        self.generate_1h = tk.BooleanVar(value=True)
        self.switch_1h = ctk.CTkSwitch(root, text="Generate 1h Video", variable=self.generate_1h)
        self.switch_1h.pack(pady=5)
        
        self.output_format_var = tk.StringVar(value="mp4")
        format_frame = ctk.CTkFrame(root)
        format_frame.pack(pady=5)
        ctk.CTkLabel(format_frame, text="Output Format:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(format_frame, variable=self.output_format_var, values=["mp4", "avi", "mkv"]).pack(side=tk.LEFT)
        
        music_frame = ctk.CTkFrame(root)
        music_frame.pack(pady=5)
        self.music_label = ctk.CTkLabel(music_frame, text="No music selected")
        self.music_label.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_frame, text="Select Music", command=self.select_music).pack(side=tk.LEFT)
        self.music_path = None
        
        self.settings_button = ctk.CTkButton(root, text="Settings & Preview", command=self.open_settings)
        self.settings_button.pack(pady=5)
        
        self.browse_button = ctk.CTkButton(root, text="Browse", command=self.browse_files)
        self.browse_button.pack(pady=5)
        
        self.start_button = ctk.CTkButton(root, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=5)
        
        self.progress = ctk.CTkProgressBar(root, width=300)
        self.progress.pack(pady=10)
        self.progress.set(0)
        
        self.current_task_label = ctk.CTkLabel(root, text="Current Task: N/A")
        self.current_task_label.pack(pady=5)
        
        self.time_label = ctk.CTkLabel(root, text="Estimated Time Remaining: N/A")
        self.time_label.pack(pady=5)
        
        self.cancel_button = ctk.CTkButton(root, text="Cancel", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(pady=5)
        
        self.output_label = ctk.CTkLabel(root, text="Output Files:")
        self.output_label.pack(pady=10)
        
        self.output_frame = ctk.CTkFrame(root)
        self.output_frame.pack(pady=5)
        
        self.input_files = []
        self.cancel_event = threading.Event()
        self.queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=1)
        self.start_time = None
        self.preview_image = None
        self.blank_ctk_image = ctk.CTkImage(light_image=Image.new('RGB', (200, 150), (0, 0, 0)), 
                                          dark_image=Image.new('RGB', (200, 150), (0, 0, 0)), size=(200, 150))
        self.root.after(50, self.process_queue)
        
        self.motion_threshold = 3000
        self.white_threshold = 200
        self.black_threshold = 50
        self.clip_limit = 1.0
        self.saturation_multiplier = 1.1
        
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.motion_threshold = int(settings.get("motion_threshold", 3000))
                self.white_threshold = int(settings.get("white_threshold", 200))
                self.black_threshold = int(settings.get("black_threshold", 50))
                self.clip_limit = float(settings.get("clip_limit", 1.0))
                self.saturation_multiplier = float(settings.get("saturation_multiplier", 1.1))
            log_session("Loaded settings from settings.json")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            logging.warning("Could not load settings, using defaults")
            log_session("Could not load settings, using defaults")
        
        self.presets = {}
        try:
            with open("presets.json", "r") as f:
                self.presets = json.load(f)
            log_session("Loaded presets from presets.json")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.info("No presets found")
            log_session("No presets found")

    def select_music(self):
        """Handle music file selection."""
        music_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if music_path:
            self.music_path = music_path
            self.music_label.configure(text=os.path.basename(music_path))
            log_session(f"Selected music file: {music_path}")
        else:
            self.music_path = None
            self.music_label.configure(text="No music selected")
            log_session("No music file selected")
    
    def toggle_theme(self, theme):
        """Switch between light and dark themes."""
        ctk.set_appearance_mode(theme)
        log_session(f"Theme changed to {theme}")
    
    def open_settings(self):
        """Open settings window with preview."""
        self.settings_window = ctk.CTkToplevel(self.root)
        self.settings_window.title("Settings & Preview")
        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_settings_close)
        log_session("Opened settings and preview window")
        
        settings_frame = ctk.CTkFrame(self.settings_window)
        settings_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        ctk.CTkLabel(settings_frame, text="Motion Sensitivity").pack(pady=5)
        self.motion_slider = ctk.CTkSlider(settings_frame, from_=500, to=20000, number_of_steps=195, command=self.update_settings)
        self.motion_slider.set(self.motion_threshold)
        self.motion_slider.pack(pady=5)
        self.motion_value_label = ctk.CTkLabel(settings_frame, text=f"Threshold: {self.motion_threshold}")
        self.motion_value_label.pack(pady=5)
        
        ctk.CTkLabel(settings_frame, text="White Threshold").pack(pady=2)
        self.white_slider = ctk.CTkSlider(settings_frame, from_=100, to=255, number_of_steps=155, command=self.update_settings)
        self.white_slider.set(self.white_threshold)
        self.white_slider.pack(pady=2)
        self.white_value_label = ctk.CTkLabel(settings_frame, text=f"White: {self.white_threshold}")
        self.white_value_label.pack(pady=2)
        
        ctk.CTkLabel(settings_frame, text="Black Threshold").pack(pady=2)
        self.black_slider = ctk.CTkSlider(settings_frame, from_=0, to=100, number_of_steps=100, command=self.update_settings)
        self.black_slider.set(self.black_threshold)
        self.black_slider.pack(pady=2)
        self.black_value_label = ctk.CTkLabel(settings_frame, text=f"Black: {self.black_threshold}")
        self.black_value_label.pack(pady=2)
        
        ctk.CTkLabel(settings_frame, text="CLAHE Clip Limit").pack(pady=2)
        self.clip_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=5.0, number_of_steps=90, command=self.update_settings)
        self.clip_slider.set(self.clip_limit)
        self.clip_slider.pack(pady=2)
        self.clip_value_label = ctk.CTkLabel(settings_frame, text=f"Clip Limit: {self.clip_limit:.1f}")
        self.clip_value_label.pack(pady=2)
        
        ctk.CTkLabel(settings_frame, text="Saturation Multiplier").pack(pady=2)
        self.saturation_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=2.0, number_of_steps=150, command=self.update_settings)
        self.saturation_slider.set(self.saturation_multiplier)
        self.saturation_slider.pack(pady=2)
        self.saturation_value_label = ctk.CTkLabel(settings_frame, text=f"Saturation: {self.saturation_multiplier:.1f}")
        self.saturation_value_label.pack(pady=2)
        
        preset_frame = ctk.CTkFrame(settings_frame)
        preset_frame.pack(pady=10)
        ctk.CTkLabel(preset_frame, text="Preset Management").pack(pady=5)
        
        self.preset_combobox = ctk.CTkComboBox(preset_frame, values=list(self.presets.keys()))
        self.preset_combobox.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Load Preset", command=self.load_preset).pack(pady=5)
        
        self.preset_name_entry = ctk.CTkEntry(preset_frame, placeholder_text="Enter preset name")
        self.preset_name_entry.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Save Preset", command=self.save_preset).pack(pady=5)
        
        ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        ctk.CTkButton(settings_frame, text="Reset to Default", command=self.reset_to_default).pack(pady=10)
        
        self.preview_frame = ctk.CTkFrame(self.settings_window)
        self.preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Select a video to enable preview")
        self.preview_label.pack(pady=5)
        
        control_frame = ctk.CTkFrame(self.preview_frame)
        control_frame.pack(pady=5)
        self.play_button = ctk.CTkButton(control_frame, text="Play Preview", command=self.start_preview_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ctk.CTkButton(control_frame, text="Stop Preview", command=self.stop_preview_playback, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.preview_cap = None
        self.playback_thread = None
        self.preview_running = threading.Event()
        if self.input_files:
            self.preview_cap = cv2.VideoCapture(self.input_files[0])
            self.fps = self.preview_cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            ctk.CTkLabel(settings_frame, text=f"Video duration: {duration:.2f} seconds").pack(pady=5)
            
            start_time_frame = ctk.CTkFrame(settings_frame)
            start_time_frame.pack(pady=2)
            ctk.CTkLabel(start_time_frame, text="Start time (s):").pack(side=tk.LEFT)
            self.start_time_entry = ctk.CTkEntry(start_time_frame, placeholder_text="0")
            self.start_time_entry.pack(side=tk.LEFT)
            
            end_time_frame = ctk.CTkFrame(settings_frame)
            end_time_frame.pack(pady=2)
            ctk.CTkLabel(end_time_frame, text="End time (s):").pack(side=tk.LEFT)
            self.end_time_entry = ctk.CTkEntry(end_time_frame, placeholder_text=f"{duration:.2f}")
            self.end_time_entry.pack(side=tk.LEFT)
            self.start_preview_playback()
    
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
        log_session(f"Updated settings: Motion Threshold={self.motion_threshold}, White Threshold={self.white_threshold}, "
                   f"Black Threshold={self.black_threshold}, Clip Limit={self.clip_limit}, Saturation Multiplier={self.saturation_multiplier}")
    
    def start_preview_playback(self):
        """Start video preview."""
        if not self.preview_cap or not self.preview_cap.isOpened():
            self.preview_label.configure(text="Select a video to enable preview", image=None)
            log_session("Cannot start preview: No video selected or video cannot be opened")
            return
        if not self.preview_running.is_set():
            start_time_str = self.start_time_entry.get()
            end_time_str = self.end_time_entry.get()
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            start_time = float(start_time_str) if start_time_str else 0
            end_time = float(end_time_str) if end_time_str else duration
            start_frame = max(0, min(int(start_time * self.fps), self.total_frames - 1))
            end_frame = max(start_frame + 1, min(int(end_time * self.fps), self.total_frames))
            self.start_frame = start_frame
            self.end_frame = end_frame
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.preview_running.set()
            self.play_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.playback_thread = threading.Thread(target=self.playback_loop)
            self.playback_thread.start()
            self.update_preview()
            log_session(f"Started preview playback from frame {start_frame} to {end_frame}")
    
    def stop_preview_playback(self):
        """Stop video preview."""
        if self.preview_running.is_set():
            self.preview_running.clear()
            if self.playback_thread:
                self.playback_thread.join()
            self.play_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.preview_label.configure(image=self.blank_ctk_image, text="Preview stopped")
            log_session("Stopped preview playback")
    
    def playback_loop(self):
        """Playback loop for preview."""
        while self.preview_running.is_set():
            if self.preview_cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.end_frame:
                self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, frame = self.preview_cap.read()
            if not ret:
                break
            normalized = normalize_frame(frame, self.clip_limit, self.saturation_multiplier)
            if normalized is not None:
                if is_white_or_black_frame(normalized, self.white_threshold, self.black_threshold):
                    with thread_lock:
                        try:
                            self.frame_queue.put(None, block=False)
                        except queue.Full:
                            self.frame_queue.get()
                            self.frame_queue.put(None)
                else:
                    frame_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
                    with thread_lock:
                        try:
                            self.frame_queue.put(frame_rgb, block=False)
                        except queue.Full:
                            self.frame_queue.get()
                            self.frame_queue.put(frame_rgb)
            time.sleep(1 / 20)
    
    def update_preview(self):
        """Update preview image."""
        if not self.preview_running.is_set():
            return
        try:
            item = self.frame_queue.get_nowait()
            if item is None:
                self.preview_label.configure(image=self.blank_ctk_image, text="Frame filtered out")
            else:
                pil_img = Image.fromarray(item)
                self.preview_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(200, 150))
                self.preview_label.configure(image=self.preview_image, text="")
        except queue.Empty:
            pass
        self.root.after(50, self.update_preview)
    
    def on_settings_close(self):
        """Handle settings window close."""
        self.stop_preview_playback()
        if self.preview_cap:
            self.preview_cap.release()
        self.settings_window.destroy()
        log_session("Closed settings and preview window")
    
    def save_preset(self):
        """Save current settings as preset."""
        preset_name = self.preset_name_entry.get().strip()
        if not preset_name:
            messagebox.showwarning("Warning", "Enter a preset name.")
            log_session("Failed to save preset: No preset name entered")
            return
        if preset_name in self.presets:
            messagebox.showwarning("Warning", "Preset name exists.")
            log_session(f"Failed to save preset: Preset '{preset_name}' already exists")
            return
        
        self.presets[preset_name] = {
            "motion_threshold": self.motion_threshold,
            "white_threshold": self.white_threshold,
            "black_threshold": self.black_threshold,
            "clip_limit": self.clip_limit,
            "saturation_multiplier": self.saturation_multiplier
        }
        with open("presets.json", "w") as f:
            json.dump(self.presets, f)
        self.preset_combobox.configure(values=list(self.presets.keys()))
        messagebox.showinfo("Info", f"Preset '{preset_name}' saved.")
        log_session(f"Saved preset '{preset_name}'")
    
    def load_preset(self):
        """Load selected preset."""
        preset = self.preset_combobox.get()
        if preset in self.presets:
            settings = self.presets[preset]
            self.motion_slider.set(settings["motion_threshold"])
            self.white_slider.set(settings["white_threshold"])
            self.black_slider.set(settings["black_threshold"])
            self.clip_slider.set(settings["clip_limit"])
            self.saturation_slider.set(settings["saturation_multiplier"])
            self.update_settings(0)
            messagebox.showinfo("Info", f"Preset '{preset}' loaded.")
            log_session(f"Loaded preset '{preset}'")
    
    def save_settings(self):
        """Save settings to file."""
        self.motion_threshold = int(self.motion_slider.get())
        self.white_threshold = int(self.white_slider.get())
        self.black_threshold = int(self.black_slider.get())
        self.clip_limit = float(self.clip_slider.get())
        self.saturation_multiplier = float(self.saturation_slider.get())
        
        settings = {
            "motion_threshold": self.motion_threshold,
            "white_threshold": self.white_threshold,
            "black_threshold": self.black_threshold,
            "clip_limit": self.clip_limit,
            "saturation_multiplier": self.saturation_multiplier
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        self.stop_preview_playback()
        if self.preview_cap:
            self.preview_cap.release()
        self.settings_window.destroy()
        log_session("Saved settings and closed settings window")
    
    def reset_to_default(self):
        """Reset settings to defaults."""
        self.motion_slider.set(3000)
        self.white_slider.set(200)
        self.black_slider.set(50)
        self.clip_slider.set(1.0)
        self.saturation_slider.set(1.1)
        self.update_settings(0)
        log_session("Reset settings to default values")
    
    def browse_files(self):
        """Handle file selection."""
        self.input_files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.input_files:
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            log_session(f"Selected {len(self.input_files)} file(s): {', '.join(self.input_files)}")
        else:
            log_session("No files selected")
    
    def start_processing(self):
        """Initiate video processing."""
        logging.info("start_processing called")
        log_session("Starting video processing")
        if not self.input_files:
            logging.warning("No files selected")
            log_session("Warning: No files selected")
            messagebox.showwarning("Warning", "No files selected.")
            return
        if not (self.generate_60s.get() or self.generate_12min.get() or self.generate_1h.get()):
            logging.warning("No video generation options selected")
            log_session("Warning: No video generation options selected")
            messagebox.showwarning("Warning", "Select at least one video to generate.")
            return
        logging.info("Clearing output frame")
        log_session("Clearing output frame")
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        logging.info("Disabling UI elements")
        log_session("Disabling UI elements")
        self.switch_60s.configure(state="disabled")
        self.switch_12min.configure(state="disabled")
        self.switch_1h.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.cancel_event.clear()
        self.start_time = time.time()
        threading.Thread(target=self.process_video_thread).start()
        log_session("Started processing thread")
    
    def process_video_thread(self):
        """Process videos in a separate thread."""
        try:
            logging.info("process_video_thread started")
            log_session("Processing thread started")
            selected_videos = []
            if self.generate_60s.get():
                selected_videos.append(("Generate 60s Video", 60))
            if self.generate_12min.get():
                selected_videos.append(("Generate 12min Video", 720))
            if self.generate_1h.get():
                selected_videos.append(("Generate 1h Video", 3600))
            log_session(f"Selected video durations: {[name for name, _ in selected_videos]}")
            
            output_format = self.output_format_var.get()
            for input_file in self.input_files:
                base, _ = os.path.splitext(input_file)
                output_files = {}
                
                for task_name, duration in selected_videos:
                    if self.cancel_event.is_set():
                        self.queue.put(("canceled", "User Cancellation"))
                        logging.info("Processing canceled by user")
                        log_session("Processing canceled by user")
                        break
                    output_file = f"{base}_{task_name.split()[1]}.{output_format}"
                    logging.info(f"Starting task: {task_name} for {input_file}")
                    log_session(f"Starting task: {task_name} for {input_file}")
                    self.queue.put(("task_start", f"{task_name} - {os.path.basename(input_file)}", 0))
                    
                    def progress_callback(progress, current, total, remaining):
                        logging.info(f"Progress: {progress:.2f}%, {current}/{total}, remaining: {remaining:.2f}s")
                        with thread_lock:
                            self.queue.put(("progress", progress, current, total, remaining))
                    
                    def status_callback(status):
                        logging.info(f"Status update: {status}")
                        self.queue.put(("status", status))
                    
                    error = process_video_multi_pass(
                        input_file, output_file, duration,
                        motion_threshold=self.motion_threshold,
                        sample_interval=5,
                        white_threshold=self.white_threshold,
                        black_threshold=self.black_threshold,
                        clip_limit=self.clip_limit,
                        saturation_multiplier=self.saturation_multiplier,
                        output_format=output_format,
                        progress_callback=progress_callback,
                        cancel_event=self.cancel_event,
                        music_path=self.music_path,
                        status_callback=status_callback
                    )
                    
                    if error:
                        self.queue.put(("canceled", error))
                        logging.error(f"Task failed: {error}")
                        log_session(f"Task failed: {error}")
                        break
                    else:
                        output_files[task_name] = output_file
                        logging.info(f"Task completed: {task_name}")
                        log_session(f"Task completed: {task_name}, output: {output_file}")
                self.queue.put(("complete", output_files, time.time() - self.start_time))
            logging.info("process_video_thread finished")
            log_session("Processing thread finished")
        except Exception as e:
            logging.error(f"Error in process_video_thread: {str(e)}", exc_info=True)
            log_session(f"Error in processing thread: {str(e)}")
            self.queue.put(("canceled", str(e)))
    
    def cancel_processing(self):
        """Cancel ongoing processing."""
        self.cancel_event.set()
        logging.info("Cancel requested")
        log_session("Cancel processing requested")
    
    def process_queue(self):
        """Process queue messages for UI updates."""
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)
    
    def handle_message(self, message):
        """Handle queue messages to update UI."""
        msg_type, *args = message
        if msg_type == "task_start":
            task_name, progress = args
            self.current_task_label.configure(text=f"Current Task: {task_name}")
            self.progress.set(0)
            self.time_label.configure(text="Estimating time...")
            logging.info(f"Task started: {task_name}")
            log_session(f"UI Update: Task started: {task_name}")
        elif msg_type == "progress":
            progress_value, current, total, remaining = args
            progress_value = min(max(progress_value, 0), 100)
            self.progress.set(progress_value / 100)
            remaining_min = remaining / 60 if remaining > 0 else 0
            self.time_label.configure(text=f"Est. Time Remaining: {remaining_min:.2f} min ({progress_value:.2f}% complete)")
            log_session(f"UI Update: Progress {progress_value:.2f}%, Est. Time Remaining: {remaining_min:.2f} min")
        elif msg_type == "status":
            status_text = args[0]
            self.current_task_label.configure(text=status_text)
            log_session(f"UI Update: Status: {status_text}")
        elif msg_type == "complete":
            output_files, elapsed = args
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
                upload_button = ctk.CTkButton(file_frame, text="Upload to YouTube")
                upload_button.configure(command=lambda f=file, t=task, b=upload_button: self.start_upload(f, t, b))
                upload_button.pack(side=tk.RIGHT, padx=5)
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Complete in {time_str}")
            self.reset_ui()
            logging.info(f"Processing completed in {time_str}")
            log_session(f"UI Update: Processing completed in {time_str}")
        elif msg_type == "canceled":
            reason = args[0]
            elapsed = time.time() - self.start_time if self.start_time else 0
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes} min {seconds} sec"
            for widget in self.output_frame.winfo_children():
                widget.destroy()
            cancel_label = ctk.CTkLabel(self.output_frame, text=f"Canceled: {reason}")
            cancel_label.pack()
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Canceled in {time_str}")
            self.reset_ui()
            logging.info(f"Processing canceled: {reason}")
            log_session(f"UI Update: Processing canceled: {reason} in {time_str}")
    
    def reset_ui(self):
        """Reset UI controls to initial state."""
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.switch_1h.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.cancel_button.configure(state="disabled")
        log_session("UI reset to initial state")
    
    def get_youtube_client(self):
        """Get authenticated YouTube API client."""
        if not hasattr(self, 'youtube_client'):
            credentials = None
            if os.path.exists('token.pickle'):
                try:
                    with open('token.pickle', 'rb') as token:
                        credentials = pickle.load(token)
                    log_session("Loaded YouTube credentials from token.pickle")
                except (pickle.PickleError, EOFError) as e:
                    logging.error(f"Failed to load token.pickle: {str(e)}")
                    log_session(f"Error: Failed to load token.pickle: {str(e)}")
                    os.remove('token.pickle')
                    credentials = None
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    try:
                        credentials.refresh(Request())
                        log_session("Refreshed YouTube credentials")
                    except Exception as e:
                        logging.error(f"Failed to refresh credentials: {str(e)}")
                        log_session(f"Error: Failed to refresh credentials: {str(e)}")
                        credentials = None
                if not credentials:
                    if not os.path.exists('client_secrets.json'):
                        messagebox.showerror("Error", "client_secrets.json not found.")
                        log_session("Error: client_secrets.json not found for YouTube authentication")
                        return None
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'client_secrets.json',
                            scopes=['https://www.googleapis.com/auth/youtube.upload']
                        )
                        credentials = flow.run_local_server(port=0)
                        log_session("Authenticated with YouTube API")
                    except Exception as e:
                        logging.error(f"Failed to authenticate: {str(e)}")
                        log_session(f"Error: Failed to authenticate with YouTube API: {str(e)}")
                        messagebox.showerror("Error", "Authentication failed.")
                        return None
                with open('token.pickle', 'wb') as token:
                    pickle.dump(credentials, token)
                    log_session("Saved YouTube credentials to token.pickle")
            self.youtube_client = build('youtube', 'v3', credentials=credentials)
            logging.info("YouTube client initialized")
            log_session("YouTube client initialized")
        return self.youtube_client
    
    def start_upload(self, file_path, task_name, button):
        """Start YouTube upload in a separate thread."""
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Video file not found: {file_path}")
            button.configure(state="normal", text="Upload to YouTube")
            log_session(f"Error: Video file not found for upload: {file_path}")
            return
        if not validate_video_file(file_path):
            messagebox.showerror("Error", "Invalid or corrupted video file.")
            button.configure(state="normal", text="Upload to YouTube")
            log_session(f"Error: Invalid or corrupted video file for upload: {file_path}")
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
        """Upload video to YouTube with retry handling."""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                youtube = self.get_youtube_client()
                if not youtube:
                    return
                duration_str = task_name.split()[1]
                title = f"Bird Box Video - {duration_str}"
                description = "Uploaded via Bird Box Video Processor"
                tags = ['bird', 'nature', 'video']
                if duration_str == "60s":
                    title += " #shorts"
                    description += " #shorts"
                    tags.append('#shorts')
                request_body = {
                    'snippet': {'title': title, 'description': description, 'tags': tags, 'categoryId': '22'},
                    'status': {'privacyStatus': 'unlisted'}
                }
                media = MediaFileUpload(file_path, resumable=True, chunksize=512 * 1024)
                request = youtube.videos().insert(part='snippet,status', body=request_body, media_body=media)
                response = request.execute()
                logging.info(f"Upload successful: {file_path}")
                log_session(f"Upload successful: {file_path}, YouTube URL: https://youtu.be/{response['id']}")
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Video uploaded: https://youtu.be/{response['id']}"))
                break
            except Exception as e:
                logging.error(f"Upload error: {str(e)}", exc_info=True)
                log_session(f"Error: Upload failed: {str(e)}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Upload failed: {str(e)}"))
                break
            finally:
                self.root.after(0, lambda b=button: b.configure(state="normal", text="Upload to YouTube"))

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()
