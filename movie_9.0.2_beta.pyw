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
from multiprocessing import Pool, Event
from packaging import version
import functools

try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network stability checks will be limited.")

# Version number
VERSION = "9.0.2_beta"
UPDATE_CHANNELS = ["Stable", "Beta"]

# Create log directory
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(filename=os.path.join(log_dir, 'upload_log.txt'), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
session_log_file = os.path.join(log_dir, f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
session_logger = logging.getLogger('session')
session_handler = logging.FileHandler(session_log_file)
session_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
session_logger.addHandler(session_handler)
session_logger.setLevel(logging.INFO)

thread_lock = threading.Lock()
cancel_event = Event()

### Helper Functions

def log_session(message):
    session_logger.info(message)

def compute_motion_score(prev_frame, current_frame, threshold=30):
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.sum(diff > threshold, dtype=np.uint32)
    logging.debug(f"Motion score: {score}")
    return score

def is_white_or_black_frame(frame, white_threshold=200, black_threshold=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > white_threshold or mean_brightness < black_threshold

def normalize_frame(frame, clip_limit=1.0, saturation_multiplier=1.1):
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
        upload_speed = st.upload() / 1_000_000
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

def get_selected_indices(input_path, motion_threshold, white_threshold, black_threshold, progress_callback=None):
    global cancel_event
    logging.info(f"Starting motion detection for {input_path}")
    log_session(f"Starting motion detection for {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {input_path}")
        log_session(f"Error: Cannot open video file: {input_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Total frames to process: {total_frames}")
    log_session(f"Total frames to process: {total_frames}")
    
    frame_skip = max(1, total_frames // 100000)
    logging.info(f"Frame skip factor: {frame_skip}")
    log_session(f"Frame skip factor: {frame_skip}")
    
    prev_frame_resized = None
    selected_indices = []
    start_time = time.time()
    
    frame_idx = 0
    while frame_idx < total_frames:
        if cancel_event.is_set():
            cap.release()
            logging.info("Motion detection canceled by user")
            log_session("Motion detection canceled by user")
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to read frame {frame_idx} from {input_path}")
            log_session(f"Warning: Failed to read frame {frame_idx}")
            frame_idx += frame_skip
            continue
        
        frame_resized = cv2.resize(frame, (640, 360))
        if prev_frame_resized is not None:
            motion_score = compute_motion_score(prev_frame_resized, frame_resized)
            logging.debug(f"Frame {frame_idx}: Motion score = {motion_score}")
            if motion_score > motion_threshold and not is_white_or_black_frame(frame_resized, white_threshold, black_threshold):
                selected_indices.append(frame_idx)
        
        prev_frame_resized = frame_resized
        frame_idx += frame_skip
        
        if frame_idx % max(100, total_frames // 100) == 0 and progress_callback:
            elapsed = time.time() - start_time
            progress = (frame_idx / total_frames) * 100
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
            logging.info(f"Processed {frame_idx}/{total_frames} frames ({progress:.2f}%)")
            log_session(f"Processed {frame_idx}/{total_frames} frames")
            progress_callback(progress, frame_idx, total_frames, remaining)
    
    if progress_callback:
        progress_callback(100, frame_idx, total_frames, 0)
    
    cap.release()
    logging.info(f"Motion detection completed for {input_path}, {len(selected_indices)} frames selected")
    log_session(f"Motion detection completed for {input_path}, {len(selected_indices)} frames selected")
    return selected_indices

def process_frame(input_path, clip_limit, saturation_multiplier, rotate, temp_dir, task):
    global cancel_event
    frame_idx, order = task
    if cancel_event.is_set():
        return None
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    normalized_frame = normalize_frame(frame, clip_limit, saturation_multiplier)
    if normalized_frame is None:
        return None
    if rotate:
        normalized_frame = cv2.rotate(normalized_frame, cv2.ROTATE_90_CLOCKWISE)
    temp_path = os.path.join(temp_dir, f"frame_{order:04d}.jpg")
    cv2.imwrite(temp_path, normalized_frame)
    return order

def generate_output_video(input_path, output_path, desired_duration, selected_indices, clip_limit=1.0, saturation_multiplier=1.1, 
                          output_format='mp4', progress_callback=None, music_paths=None, music_volume=1.0, 
                          status_callback=None, custom_ffmpeg_args=None, watermark_text=None):
    try:
        logging.info(f"Generating video: {output_path}")
        log_session(f"Generating video: {output_path}")
        if status_callback:
            status_callback("Opening video file...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {input_path}")
            log_session(f"Error: Cannot open video file: {input_path}")
            return "Failed to open video file"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames <= 0 or fps <= 0:
            logging.error(f"Invalid video properties: total_frames={total_frames}, fps={fps}")
            log_session(f"Error: Invalid video properties: total_frames={total_frames}, fps={fps}")
            return "Invalid video properties"
        
        rotate = desired_duration == 60
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            target_frames_count = int(desired_duration * fps)
            if len(selected_indices) > target_frames_count:
                step = len(selected_indices) / target_frames_count
                final_indices = [selected_indices[int(i * step)] for i in range(target_frames_count)]
            else:
                final_indices = selected_indices
            
            if status_callback:
                status_callback("Saving processed frames...")
            logging.info("Saving processed frames")
            log_session("Saving processed frames")
            frame_tasks = [(idx, i) for i, idx in enumerate(final_indices)]
            with Pool() as pool:
                partial_process_frame = functools.partial(
                    process_frame,
                    input_path,
                    clip_limit,
                    saturation_multiplier,
                    rotate,
                    temp_dir
                )
                results = pool.map(partial_process_frame, frame_tasks)
                frame_counter = len([r for r in results if r is not None])
            logging.info(f"Saved {frame_counter} frames to temporary directory")
            log_session(f"Saved {frame_counter} frames to temporary directory")
            if status_callback:
                status_callback("Frames saved")

            if frame_counter == 0:
                logging.warning("No frames saved for the final video")
                log_session("Warning: No frames saved for the final video")
                return "No frames written after processing"

            num_frames = frame_counter
            new_fps = num_frames / desired_duration if num_frames < target_frames_count else fps
            logging.info(f"Adjusting frame rate to {new_fps:.2f} fps")
            log_session(f"Adjusting frame rate to {new_fps:.2f} fps")

            if status_callback:
                status_callback("Creating video from frames...")
            logging.info("Creating video with FFmpeg")
            log_session("Creating video with FFmpeg")
            temp_final_path = f"temp_final_{uuid.uuid4().hex}.{output_format}"
            cmd = ['ffmpeg', '-framerate', str(new_fps), '-i', os.path.join(temp_dir, 'frame_%04d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p']
            if watermark_text:
                cmd.extend(['-vf', f'drawtext=text={watermark_text}:fontcolor=white:fontsize=24:x=10:y=10'])
            if custom_ffmpeg_args:
                cmd.extend(custom_ffmpeg_args)
            cmd.extend(['-y', temp_final_path])
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            logging.info("Video created from frames")
            log_session("Video created from frames")

            music_path = music_paths.get(desired_duration, music_paths.get("default")) if music_paths else None
            if music_path and os.path.exists(music_path):
                if status_callback:
                    status_callback("Adding music...")
                logging.info("Adding music with FFmpeg")
                log_session("Adding music with FFmpeg")
                cmd = [
                    'ffmpeg', '-i', temp_final_path, '-i', music_path,
                    '-filter_complex', f"[1:a]volume={music_volume}[a]",
                    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
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
                progress_callback(100, frame_counter, len(final_indices), 0)
            logging.info(f"Video generation completed: {output_path}")
            log_session(f"Video generation completed: {output_path}")
            return None

    except Exception as e:
        logging.error(f"Error in generate_output_video: {str(e)}", exc_info=True)
        log_session(f"Error: {str(e)}")
        return f"Error: {str(e)}"

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        log_session("Application started")
        self.root.resizable(True, True)
        self.root.geometry("800x600")

        self.theme_var = tk.StringVar(value="dark")
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
        ctk.CTkOptionMenu(format_frame, variable=self.output_format_var, values=["mp4", "avi", "mkv", "mov", "wmv"]).pack(side=tk.LEFT)

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

        self.output_frame = ctk.CTkScrollableFrame(root)
        self.output_frame.pack(pady=5, fill='both', expand=True)

        self.input_files = []
        self.queue = queue.Queue()
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
        self.music_volume = 1.0
        self.output_dir = None
        self.custom_ffmpeg_args = None
        self.watermark_text = None
        self.preview_running = threading.Event()
        self.update_channel = "Stable"

        self.music_paths = {"default": None, 60: None, 720: None, 3600: None}
        self.load_settings()
        self.load_presets()
        self.check_for_updates()

    def load_settings(self):
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
                loaded_music_paths = settings.get("music_paths", {})
                for key in self.music_paths:
                    if str(key) in loaded_music_paths:
                        self.music_paths[key] = loaded_music_paths[str(key)]
            log_session("Loaded settings from settings.json")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            logging.warning("Could not load settings, using defaults")
            log_session("Could not load settings, using defaults")

    def save_settings(self):
        self.motion_threshold = int(self.motion_slider.get())
        self.white_threshold = int(self.white_slider.get())
        self.black_threshold = int(self.black_slider.get())
        self.clip_limit = float(self.clip_slider.get())
        self.saturation_multiplier = float(self.saturation_slider.get())
        self.music_volume = self.music_volume_slider.get()
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
            "update_channel": self.update_channel
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        self.on_settings_close()
        log_session("Saved settings and closed settings window")

    def load_presets(self):
        try:
            with open("presets.json", "r") as f:
                self.presets = json.load(f)
            log_session("Loaded presets from presets.json")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.info("No presets found")
            log_session("No presets found")
            self.presets = {}

    def check_for_updates(self):
        try:
            channel = self.update_channel
            response = requests.get(f"https://raw.githubusercontent.com/SimSalabimse/NestCams/main/{channel}_version.txt", timeout=5)
            latest_version_str = response.text.strip()
            current_version = version.parse(VERSION)
            latest_version = version.parse(latest_version_str)
            if latest_version > current_version:
                messagebox.showinfo("Update Available", f"Version {latest_version_str} is available for {channel} channel! Please update via GitHub.")
                log_session(f"Update available for {channel}: {latest_version_str}")
        except requests.RequestException as e:
            logging.error(f"Update check failed: {e}")
            log_session(f"Update check failed: {e}")

    def toggle_theme(self, theme):
        ctk.set_appearance_mode(theme.lower())
        log_session(f"Theme changed to {theme}")

    def open_settings(self):
        self.settings_window = ctk.CTkToplevel(self.root)
        self.settings_window.title("Settings & Preview")
        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_settings_close)
        self.settings_window.resizable(True, True)
        self.settings_window.lift()
        self.settings_window.transient(self.root)
        self.settings_window.geometry("800x600")
        log_session("Opened settings and preview window")

        settings_frame = ctk.CTkScrollableFrame(self.settings_window)
        settings_frame.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

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

        update_channel_frame = ctk.CTkFrame(settings_frame)
        update_channel_frame.pack(pady=5)
        ctk.CTkLabel(update_channel_frame, text="Update Channel:").pack(side=tk.LEFT)
        self.update_channel_var = tk.StringVar(value=self.update_channel)
        ctk.CTkOptionMenu(update_channel_frame, variable=self.update_channel_var, values=UPDATE_CHANNELS).pack(side=tk.LEFT)

        output_dir_frame = ctk.CTkFrame(settings_frame)
        output_dir_frame.pack(pady=5)
        ctk.CTkLabel(output_dir_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_label = ctk.CTkLabel(output_dir_frame, text=self.output_dir or "Default")
        self.output_dir_label.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(output_dir_frame, text="Browse", command=self.select_output_dir).pack(side=tk.LEFT)

        ffmpeg_frame = ctk.CTkFrame(settings_frame)
        ffmpeg_frame.pack(pady=5)
        ctk.CTkLabel(ffmpeg_frame, text="Custom FFmpeg Args:").pack(side=tk.LEFT)
        self.ffmpeg_entry = ctk.CTkEntry(ffmpeg_frame, placeholder_text="e.g., -vf scale=1280:720")
        if self.custom_ffmpeg_args:
            self.ffmpeg_entry.insert(0, " ".join(self.custom_ffmpeg_args))
        self.ffmpeg_entry.pack(side=tk.LEFT, padx=5)

        watermark_frame = ctk.CTkFrame(settings_frame)
        watermark_frame.pack(pady=5)
        ctk.CTkLabel(watermark_frame, text="Watermark Text:").pack(side=tk.LEFT)
        self.watermark_entry = ctk.CTkEntry(watermark_frame, placeholder_text="Enter watermark")
        if self.watermark_text:
            self.watermark_entry.insert(0, self.watermark_text)
        self.watermark_entry.pack(side=tk.LEFT, padx=5)

        music_settings_frame = ctk.CTkFrame(settings_frame)
        music_settings_frame.pack(pady=10)
        ctk.CTkLabel(music_settings_frame, text="Music Settings").pack(pady=5)

        default_music_frame = ctk.CTkFrame(music_settings_frame)
        default_music_frame.pack(pady=2)
        ctk.CTkLabel(default_music_frame, text="Default Music:").pack(side=tk.LEFT)
        default_path = self.music_paths.get("default")
        self.music_label_default = ctk.CTkLabel(default_music_frame, text="No music selected" if not default_path else os.path.basename(default_path))
        self.music_label_default.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(default_music_frame, text="Select", command=self.select_music_default).pack(side=tk.LEFT)

        music_60s_frame = ctk.CTkFrame(music_settings_frame)
        music_60s_frame.pack(pady=2)
        ctk.CTkLabel(music_60s_frame, text="Music for 60s Video:").pack(side=tk.LEFT)
        path_60s = self.music_paths.get(60)
        self.music_label_60s = ctk.CTkLabel(music_60s_frame, text="No music selected" if not path_60s else os.path.basename(path_60s))
        self.music_label_60s.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_60s_frame, text="Select", command=self.select_music_60s).pack(side=tk.LEFT)

        music_12min_frame = ctk.CTkFrame(music_settings_frame)
        music_12min_frame.pack(pady=2)
        ctk.CTkLabel(music_12min_frame, text="Music for 12min Video:").pack(side=tk.LEFT)
        path_12min = self.music_paths.get(720)
        self.music_label_12min = ctk.CTkLabel(music_12min_frame, text="No music selected" if not path_12min else os.path.basename(path_12min))
        self.music_label_12min.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_12min_frame, text="Select", command=self.select_music_12min).pack(side=tk.LEFT)

        music_1h_frame = ctk.CTkFrame(music_settings_frame)
        music_1h_frame.pack(pady=2)
        ctk.CTkLabel(music_1h_frame, text="Music for 1h Video:").pack(side=tk.LEFT)
        path_1h = self.music_paths.get(3600)
        self.music_label_1h = ctk.CTkLabel(music_1h_frame, text="No music selected" if not path_1h else os.path.basename(path_1h))
        self.music_label_1h.pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(music_1h_frame, text="Select", command=self.select_music_1h).pack(side=tk.LEFT)

        volume_frame = ctk.CTkFrame(music_settings_frame)
        volume_frame.pack(pady=2)
        ctk.CTkLabel(volume_frame, text="Music Volume (0.0 - 1.0):").pack(side=tk.LEFT)
        self.music_volume_slider = ctk.CTkSlider(volume_frame, from_=0.0, to=1.0, number_of_steps=100, command=lambda v: setattr(self, 'music_volume', v))
        self.music_volume_slider.set(self.music_volume)
        self.music_volume_slider.pack(side=tk.LEFT)

        schedule_frame = ctk.CTkFrame(settings_frame)
        schedule_frame.pack(pady=10)
        ctk.CTkLabel(schedule_frame, text="Schedule Processing (HH:MM):").pack(pady=2)
        self.schedule_entry = ctk.CTkEntry(schedule_frame, placeholder_text="e.g., 14:30")
        self.schedule_entry.pack(pady=2)
        ctk.CTkButton(schedule_frame, text="Set Schedule", command=self.set_schedule).pack(pady=2)

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
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Select a video to enable preview", image=self.blank_ctk_image)
        self.preview_label.pack(pady=5)

        control_frame = ctk.CTkFrame(self.preview_frame)
        control_frame.pack(pady=5)
        self.preview_button = ctk.CTkButton(control_frame, text="Start Preview", command=self.toggle_preview, state="disabled")
        self.preview_button.pack(side=tk.LEFT, padx=5)
        self.preview_slider = ctk.CTkSlider(control_frame, from_=0, to=0, number_of_steps=0, command=self.update_preview_frame)
        self.preview_slider.pack(side=tk.LEFT, padx=5)

        self.preview_cap = None
        self.playback_thread = None
        if self.input_files:
            self.initialize_preview()

    def initialize_preview(self):
        self.preview_cap = cv2.VideoCapture(self.input_files[0])
        if self.preview_cap.isOpened():
            self.fps = self.preview_cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.preview_slider.configure(to=self.total_frames - 1, number_of_steps=self.total_frames - 1)
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            ctk.CTkLabel(self.settings_window, text=f"Video duration: {duration:.2f} seconds").pack(pady=5)
            self.preview_button.configure(state="normal")

    def toggle_preview(self):
        if self.preview_running.is_set():
            self.stop_preview_playback()
        else:
            self.start_preview_playback()

    def start_preview_playback(self):
        if self.preview_cap and self.preview_cap.isOpened() and not self.preview_running.is_set():
            self.preview_running.set()
            self.preview_button.configure(text="Stop Preview")
            self.playback_thread = threading.Thread(target=self.preview_playback)
            self.playback_thread.start()
            log_session("Started preview playback")

    def stop_preview_playback(self):
        if self.preview_running.is_set():
            self.preview_running.clear()
            if self.playback_thread:
                self.playback_thread.join()
                self.playback_thread = None
            self.preview_button.configure(text="Start Preview")
            self.preview_label.configure(image=self.blank_ctk_image)
            log_session("Stopped preview playback")

    def preview_playback(self):
        if not self.preview_cap or not self.preview_cap.isOpened():
            return
        start_frame = int(self.preview_slider.get())
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while self.preview_running.is_set():
            ret, frame = self.preview_cap.read()
            if not ret:
                self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                continue
            frame = cv2.resize(frame, (200, 150))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
            self.preview_label.configure(image=ctk_img)
            self.preview_image = ctk_img
            self.preview_slider.set(self.preview_cap.get(cv2.CAP_PROP_POS_FRAMES))
            time.sleep(1 / self.fps if self.fps > 0 else 0.033)

    def update_preview_frame(self, frame_idx):
        if self.preview_cap and not self.preview_running.is_set():
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = self.preview_cap.read()
            if ret:
                frame = cv2.resize(frame, (200, 150))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
                self.preview_label.configure(image=ctk_img)
                self.preview_image = ctk_img

    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = directory
            self.output_dir_label.configure(text=os.path.basename(directory) or directory)
            log_session(f"Selected output directory: {directory}")

    def set_schedule(self):
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
        while True:
            schedule.run_pending()
            time.sleep(60)

    def on_settings_close(self):
        if self.preview_running.is_set():
            self.stop_preview_playback()
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        self.settings_window.destroy()
        log_session("Closed settings and preview window")

    def update_settings(self, value):
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
        log_session(f"Updated settings: Motion={self.motion_threshold}, White={self.white_threshold}, Black={self.black_threshold}, Clip={self.clip_limit}, Saturation={self.saturation_multiplier}")

    def select_music_default(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths["default"] = path
            self.music_label_default.configure(text=os.path.basename(path))
        else:
            self.music_paths["default"] = None
            self.music_label_default.configure(text="No music selected")

    def select_music_60s(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[60] = path
            self.music_label_60s.configure(text=os.path.basename(path))
        else:
            self.music_paths[60] = None
            self.music_label_60s.configure(text="No music selected")

    def select_music_12min(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[720] = path
            self.music_label_12min.configure(text=os.path.basename(path))
        else:
            self.music_paths[720] = None
            self.music_label_12min.configure(text="No music selected")

    def select_music_1h(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg")])
        if path:
            self.music_paths[3600] = path
            self.music_label_1h.configure(text=os.path.basename(path))
        else:
            self.music_paths[3600] = None
            self.music_label_1h.configure(text="No music selected")

    def reset_to_default(self):
        self.motion_slider.set(3000)
        self.white_slider.set(200)
        self.black_slider.set(50)
        self.clip_slider.set(1.0)
        self.saturation_slider.set(1.1)
        self.music_volume_slider.set(1.0)
        self.output_dir = None
        self.output_dir_label.configure(text="Default")
        self.ffmpeg_entry.delete(0, tk.END)
        self.watermark_entry.delete(0, tk.END)
        self.update_channel_var.set("Stable")
        self.update_settings(0)
        log_session("Reset settings to default values")

    def browse_files(self):
        self.input_files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv")])
        if self.input_files:
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            log_session(f"Selected {len(self.input_files)} file(s): {', '.join(self.input_files)}")
        else:
            log_session("No files selected")

    def start_processing(self):
        global cancel_event
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
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        self.switch_60s.configure(state="disabled")
        self.switch_12min.configure(state="disabled")
        self.switch_1h.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        cancel_event.clear()
        self.start_time = time.time()
        threading.Thread(target=self.process_video_thread).start()
        log_session("Started processing thread")

    def process_video_thread(self):
        global cancel_event
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
            total_tasks = len(self.input_files) * (len(selected_videos) + 1)
            task_count = 0
            has_error = False

            for input_file in self.input_files:
                base, _ = os.path.splitext(input_file)
                output_files = {}

                self.queue.put(("task_start", f"Motion Detection - {os.path.basename(input_file)}", task_count / total_tasks * 100))
                task_count += 1
                def motion_progress_callback(progress, current, total, remaining):
                    self.queue.put(("progress", progress, current, total, remaining))

                selected_indices = get_selected_indices(
                    input_file, self.motion_threshold, self.white_threshold, self.black_threshold,
                    progress_callback=motion_progress_callback
                )
                if selected_indices is None:
                    self.queue.put(("canceled", "Processing canceled by user"))
                    has_error = True
                    break
                if not selected_indices:
                    logging.warning(f"No frames selected for {input_file}")
                    log_session(f"Warning: No frames selected for {input_file}")
                    self.queue.put(("canceled", "No frames selected after motion detection"))
                    has_error = True
                    continue

                for task_name, duration in selected_videos:
                    if cancel_event.is_set():
                        self.queue.put(("canceled", "User Cancellation"))
                        has_error = True
                        break
                    output_file = f"{base}_{task_name.split()[1]}.{output_format}"
                    if self.output_dir:
                        output_file = os.path.join(self.output_dir, os.path.basename(output_file))
                    self.queue.put(("task_start", f"{task_name} - {os.path.basename(input_file)}", task_count / total_tasks * 100))
                    task_count += 1

                    def progress_callback(progress, current, total, remaining):
                        logging.info(f"Progress: {progress:.2f}%, {current}/{total}, remaining: {remaining:.2f}s")
                        with thread_lock:
                            self.queue.put(("progress", progress, current, total, remaining))

                    def status_callback(status):
                        logging.info(f"Status update: {status}")
                        self.queue.put(("status", status))

                    error = generate_output_video(
                        input_file, output_file, duration, selected_indices,
                        clip_limit=self.clip_limit,
                        saturation_multiplier=self.saturation_multiplier,
                        output_format=output_format,
                        progress_callback=progress_callback,
                        music_paths=self.music_paths,
                        music_volume=self.music_volume,
                        status_callback=status_callback,
                        custom_ffmpeg_args=self.custom_ffmpeg_args,
                        watermark_text=self.watermark_text
                    )
                    if error:
                        self.queue.put(("canceled", error))
                        has_error = True
                        break
                    else:
                        output_files[task_name] = output_file
                if not has_error:
                    self.queue.put(("complete", output_files, time.time() - self.start_time))
            logging.info("process_video_thread finished")
            log_session("Processing thread finished")
        except Exception as e:
            logging.error(f"Error in process_video_thread: {str(e)}", exc_info=True)
            log_session(f"Error in processing thread: {str(e)}")
            self.queue.put(("canceled", str(e)))

    def process_queue(self):
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                msg_type, *args = message
                if msg_type == "task_start":
                    task, percentage = args
                    self.current_task_label.configure(text=f"Current Task: {task}")
                    self.progress.set(percentage / 100)
                elif msg_type == "progress":
                    progress, current, total, remaining = args
                    self.progress.set(progress / 100)
                    self.time_label.configure(text=f"Estimated Time Remaining: {remaining:.2f}s")
                elif msg_type == "status":
                    status = args[0]
                    self.current_task_label.configure(text=f"Current Task: {status}")
                elif msg_type == "complete":
                    output_files, elapsed = args
                    for task_name, file_path in output_files.items():
                        ctk.CTkLabel(self.output_frame, text=f"{task_name}: {file_path}").pack(pady=2)
                    self.progress.set(1.0)
                    self.time_label.configure(text=f"Completed in {elapsed:.2f}s")
                    self.reset_ui()
                elif msg_type == "canceled":
                    reason = args[0]
                    ctk.CTkLabel(self.output_frame, text=f"Processing canceled: {reason}").pack(pady=2)
                    self.progress.set(0)
                    self.time_label.configure(text="Estimated Time Remaining: N/A")
                    self.reset_ui()
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)

    def reset_ui(self):
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.switch_1h.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.cancel_button.configure(state="disabled")
        self.current_task_label.configure(text="Current Task: N/A")

    def cancel_processing(self):
        global cancel_event
        cancel_event.set()
        log_session("User canceled processing")

    def load_preset(self):
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

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()
