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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from packaging import version
import functools
import psutil
import httplib2
import google_auth_httplib2

try:
    import speedtest
except ImportError:
    speedtest = None
    logging.warning("speedtest module not found. Network stability checks will be limited.")

VERSION = "9.0.4_beta"
UPDATE_CHANNELS = ["Stable", "Beta"]

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'upload_log.txt'), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
session_log_file = os.path.join(log_dir, f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
session_logger = logging.getLogger('session')
session_handler = logging.FileHandler(session_log_file)
session_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
session_logger.addHandler(session_handler)
session_logger.setLevel(logging.INFO)

thread_lock = threading.Lock()
cancel_event = Event()

# Global variables for dynamic configuration
FRAME_SIZE = (640, 360)
BATCH_SIZE = 4
WORKER_PROCESSES = 2

def log_session(message):
    session_logger.info(message)

def check_system_specs():
    global FRAME_SIZE, BATCH_SIZE, WORKER_PROCESSES
    cpu_cores = os.cpu_count() or 1  # Default to 1 if None
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB

    if total_ram_gb < 8:
        FRAME_SIZE = (320, 180)
        BATCH_SIZE = max(1, cpu_cores // 2)
        WORKER_PROCESSES = 1
    elif total_ram_gb < 16:
        FRAME_SIZE = (640, 360)
        BATCH_SIZE = max(2, cpu_cores // 2)
        WORKER_PROCESSES = min(2, cpu_cores)
    else:
        FRAME_SIZE = (1280, 720)
        BATCH_SIZE = max(4, cpu_cores)
        WORKER_PROCESSES = min(4, cpu_cores)

    logging.info(f"System specs: CPU cores={cpu_cores}, RAM={total_ram_gb:.2f} GB")
    logging.info(f"Configured: Frame size={FRAME_SIZE}, Batch size={BATCH_SIZE}, Workers={WORKER_PROCESSES}")
    log_session(f"Configured based on system specs: Frame size={FRAME_SIZE}, Batch size={BATCH_SIZE}, Workers={WORKER_PROCESSES}")

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
        available_memory = psutil.virtual_memory().available
        frame_size = frame.nbytes
        if frame_size > available_memory * 0.5:
            logging.warning(f"Frame size {frame_size} exceeds 50% of available memory {available_memory}")
            return None

        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
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
    except Exception as e:
        logging.error(f"Error in normalize_frame: {str(e)}")
        log_session(f"Error in normalize_frame: {str(e)}")
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
    except Exception as e:
        logging.warning(f"Network ping failed: {str(e)}")
        log_session(f"Network ping failed: {str(e)}")
        return False
    
    if speedtest is None:
        logging.info("No speedtest module, relying on ping")
        log_session("No speedtest module, relying on ping")
        return True
    
    def run_speedtest():
        st = speedtest.Speedtest()
        st.get_best_server()
        return st.upload() / 1_000_000
    
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_speedtest)
            upload_speed = future.result(timeout=30)
        if upload_speed < 1.0:
            logging.warning(f"Upload speed too low: {upload_speed:.2f} Mbps")
            log_session(f"Upload speed too low: {upload_speed:.2f} Mbps")
            return False
        logging.info(f"Network stable: Upload speed {upload_speed:.2f} Mbps")
        log_session(f"Network stable: Upload speed {upload_speed:.2f} Mbps")
        return True
    except FuturesTimeoutError:
        logging.warning("Speed test timed out, relying on ping")
        log_session("Speed test timed out, relying on ping")
        return True
    except Exception as e:
        logging.warning(f"Speed test failed: {str(e)}, relying on ping")
        log_session(f"Speed test failed: {str(e)}, relying on ping")
        return True

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
    prev_frame_resized = None
    selected_indices = []
    start_time = time.time()
    for frame_idx in range(total_frames):
        if cancel_event.is_set():
            cap.release()
            logging.info("Motion detection canceled by user")
            log_session("Motion detection canceled by user")
            return None
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to read frame {frame_idx} from {input_path}")
            break
        frame_resized = cv2.resize(frame, FRAME_SIZE)
        if prev_frame_resized is not None:
            motion_score = compute_motion_score(prev_frame_resized, frame_resized)
            if motion_score > motion_threshold and not is_white_or_black_frame(frame_resized, white_threshold, black_threshold):
                selected_indices.append(frame_idx)
        prev_frame_resized = frame_resized
        if frame_idx % 100 == 0 and progress_callback:
            elapsed = time.time() - start_time
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
            progress = (frame_idx / total_frames) * 100
            progress_callback(progress, frame_idx, total_frames, remaining)
    cap.release()
    logging.info(f"Motion detection completed for {input_path}, {len(selected_indices)} frames selected")
    log_session(f"Motion detection completed for {input_path}, {len(selected_indices)} frames selected")
    return selected_indices

def process_frame_batch(input_path, clip_limit, saturation_multiplier, rotate, temp_dir, tasks):
    global cancel_event
    if cancel_event.is_set():
        return []
    results = []
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file in process_frame_batch: {input_path}")
        return results
    for frame_idx, order in tasks:
        if cancel_event.is_set():
            cap.release()
            return results
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        normalized_frame = normalize_frame(frame, clip_limit, saturation_multiplier)
        if normalized_frame is None:
            continue
        if rotate:
            normalized_frame = cv2.rotate(normalized_frame, cv2.ROTATE_90_CLOCKWISE)
        temp_path = os.path.join(temp_dir, f"frame_{order:04d}.jpg")
        cv2.imwrite(temp_path, normalized_frame)
        results.append(order)
    cap.release()
    return results

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
            return "Failed to open video file", 0, 0, 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames <= 0 or fps <= 0:
            logging.error(f"Invalid video properties: total_frames={total_frames}, fps={fps}")
            log_session(f"Error: Invalid video properties: total_frames={total_frames}, fps={fps}")
            return "Invalid video properties", 0, 0, 0
        
        rotate = desired_duration <= 60
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
            task_batches = [frame_tasks[i:i + BATCH_SIZE] for i in range(0, len(frame_tasks), BATCH_SIZE)]
            
            with Pool(processes=WORKER_PROCESSES) as pool:
                partial_process_frame_batch = functools.partial(
                    process_frame_batch,
                    input_path,
                    clip_limit,
                    saturation_multiplier,
                    rotate,
                    temp_dir
                )
                results = pool.map(partial_process_frame_batch, task_batches)
                frame_counter = sum(len(batch) for batch in results)
            
            logging.info(f"Saved {frame_counter} frames to temporary directory")
            log_session(f"Saved {frame_counter} frames to temporary directory")
            if status_callback:
                status_callback("Frames saved")

            if frame_counter == 0:
                logging.warning("No frames saved for the final video")
                log_session("Warning: No frames saved for the final video")
                return "No frames written after processing", 0, 0, 0

            num_frames = frame_counter
            new_fps = num_frames / desired_duration if num_frames < target_frames_count else fps
            logging.info(f"Adjusting frame rate to {new_fps:.2f} fps")
            log_session(f"Adjusting frame rate to {new_fps:.2f} fps")

            if status_callback:
                status_callback("Creating video from frames...")
            logging.info("Creating video with FFmpeg")
            log_session("Creating video with FFmpeg")
            temp_final_path = f"temp_final_{uuid.uuid4().hex}.{output_format}"
            
            cmd = ['ffmpeg', '-framerate', str(new_fps), '-i', os.path.join(temp_dir, 'frame_%04d.jpg')]
            try:
                subprocess.run(['ffmpeg', '-hwaccels'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
            except subprocess.CalledProcessError:
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
            cmd.extend(['-pix_fmt', 'yuv420p'])
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
            return None, frame_counter, len(selected_indices), time.time() - start_time

    except Exception as e:
        logging.error(f"Error in generate_output_video: {str(e)}", exc_info=True)
        log_session(f"Error: {str(e)}")
        return f"Error: {str(e)}", 0, 0, 0

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        log_session("Application started")
        self.root.resizable(True, True)
        self.root.geometry("800x600")

        check_system_specs()  # Configure settings based on system specs

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

        custom_frame = ctk.CTkFrame(root)
        custom_frame.pack(pady=5)
        ctk.CTkLabel(custom_frame, text="Custom Duration (seconds):").pack(side=tk.LEFT, padx=5)
        self.custom_duration_entry = ctk.CTkEntry(custom_frame, placeholder_text="e.g., 120")
        self.custom_duration_entry.pack(side=tk.LEFT, padx=5)

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
        self.preview_queue = queue.Queue(maxsize=5)
        self.start_time = None
        self.preview_image = None
        self.blank_ctk_image = ctk.CTkImage(light_image=Image.new('RGB', (200, 150), (0, 0, 0)), 
                                            dark_image=Image.new('RGB', (200, 150), (0, 0, 0)), size=(200, 150))
        self.root.after(50, self.process_queue)
        self.root.after(33, self.update_preview)

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
            url = f"https://raw.githubusercontent.com/SimSalabimse/NestCams/main/{channel}_version.txt"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            latest_version_str = response.text.strip()
            if not any(char.isdigit() for char in latest_version_str):
                raise ValueError(f"Response does not contain a valid version: '{latest_version_str}'")
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
        except requests.RequestException as e:
            logging.error(f"Failed to fetch update information from {url}: {e}")
            log_session(f"Update check failed due to network issue: {e}")
        except ValueError as e:
            logging.error(f"Invalid version data received: {e}")
            log_session(f"Update check failed due to invalid version data: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during update check: {e}")
            log_session(f"Unexpected error during update check: {e}")

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
        self.clip_slider = ctk.CTkSlider(settings_frame, from_=0.2, to=5.0, number_of_steps=96, command=self.update_settings)
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
        self.music_volume_slider = ctk.CTkSlider(volume_frame, from_=0.0, to=1.0, number_of_steps=100, command=self.update_volume_label)
        self.music_volume_slider.set(self.music_volume)
        self.music_volume_slider.pack(side=tk.LEFT)
        self.volume_value_label = ctk.CTkLabel(volume_frame, text=f"{int(self.music_volume * 100)}%")
        self.volume_value_label.pack(side=tk.LEFT, padx=5)

        schedule_frame = ctk.CTkFrame(settings_frame)
        schedule_frame.pack(pady=10)
        ctk.CTkLabel(schedule_frame, text="Schedule Processing (HH:MM):").pack(side=tk.LEFT)
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
        self.preview_slider = ctk.CTkSlider(control_frame, from_=0, to=0, number_of_steps=0, command=self.seek_preview)
        self.preview_slider.pack(side=tk.LEFT, padx=5)

        if self.input_files:
            self.initialize_preview()

    def update_volume_label(self, value):
        percentage = int(float(value) * 100)
        self.music_volume = float(value)
        self.volume_value_label.configure(text=f"{percentage}%")
        log_session(f"Music volume set to {percentage}%")

    def initialize_preview(self):
        if self.input_files and not self.preview_cap:
            self.preview_cap = cv2.VideoCapture(self.input_files[0])
            if self.preview_cap.isOpened():
                self.fps = max(self.preview_cap.get(cv2.CAP_PROP_FPS), 1)
                self.total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.preview_slider.configure(to=self.total_frames - 1, number_of_steps=self.total_frames - 1)
                duration = self.total_frames / self.fps
                ctk.CTkLabel(self.settings_window, text=f"Video duration: {duration:.2f} seconds").pack(pady=5)
                self.preview_button.configure(state="normal")
                log_session(f"Initialized preview for {self.input_files[0]}")
            else:
                logging.error(f"Failed to open video for preview: {self.input_files[0]}")
                log_session(f"Error: Failed to initialize preview for {self.input_files[0]}")
                self.preview_cap = None

    def toggle_preview(self):
        if self.preview_running:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self):
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
            frame = cv2.resize(frame, (200, 150), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
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
        if self.preview_running and self.settings_window.winfo_exists():
            try:
                ctk_img, next_frame = self.preview_queue.get_nowait()
                if self.preview_label.winfo_exists():
                    self.preview_label.configure(image=ctk_img)
                    self.preview_image = ctk_img
                    self.preview_slider.set(next_frame)
            except queue.Empty:
                pass
        self.root.after(33, self.update_preview)

    def seek_preview(self, frame_idx):
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
            else:
                log_session(f"Failed to seek to frame {frame_idx}")

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
        if self.preview_running:
            self.stop_preview()
        self.settings_window.destroy()
        log_session("Settings window closed")

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
        self.update_volume_label(1.0)
        log_session("Reset settings to default values")

    def browse_files(self):
        self.input_files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv")])
        if self.input_files:
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
            if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
                self.initialize_preview()
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
        if not (self.generate_60s.get() or self.generate_12min.get() or self.generate_1h.get() or self.custom_duration_entry.get()):
            logging.warning("No video generation options selected")
            log_session("Warning: No video generation options selected")
            messagebox.showwarning("Warning", "Select at least one video duration to generate.")
            return
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        self.switch_60s.configure(state="disabled")
        self.switch_12min.configure(state="disabled")
        self.switch_1h.configure(state="disabled")
        self.custom_duration_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        cancel_event.clear()
        self.start_time = time.time()
        self.analytics_data = []
        threading.Thread(target=self.process_video_thread).start()
        log_session("Started processing thread")

    def process_single_video(self, input_file, selected_videos, output_format, total_tasks, task_count_queue):
        global cancel_event
        try:
            base, _ = os.path.splitext(input_file)
            output_files = {}
            task_count = task_count_queue.get()

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
                return None
            if not selected_indices:
                logging.warning(f"No frames selected for {input_file}")
                log_session(f"Warning: No frames selected for {input_file}")
                self.queue.put(("canceled", "No frames selected after motion detection"))
                return None

            for task_name, duration in selected_videos:
                if cancel_event.is_set():
                    self.queue.put(("canceled", "User Cancellation"))
                    return None
                output_file = f"{base}_{task_name.split()[1] if 'Generate' in task_name else duration}s.{output_format}"
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

                error, frames_processed, motion_events, proc_time = generate_output_video(
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
                    return None
                else:
                    output_files[task_name] = output_file
                    self.analytics_data.append({
                        "file": os.path.basename(input_file),
                        "duration": duration,
                        "frames_processed": frames_processed,
                        "motion_events": motion_events,
                        "processing_time": proc_time
                    })
            task_count_queue.put(task_count)
            return output_files
        except Exception as e:
            logging.error(f"Error in process_single_video: {str(e)}", exc_info=True)
            log_session(f"Error in processing {input_file}: {str(e)}")
            self.queue.put(("canceled", str(e)))
            return None

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
            custom_duration = self.custom_duration_entry.get()
            if custom_duration:
                try:
                    duration = int(custom_duration)
                    if duration > 0:
                        selected_videos.append((f"Generate {duration}s Video", duration))
                    else:
                        raise ValueError("Duration must be positive")
                except ValueError:
                    self.queue.put(("canceled", "Invalid custom duration"))
                    return

            output_format = self.output_format_var.get()
            total_tasks = len(self.input_files) * (len(selected_videos) + 1)
            task_count_queue = queue.Queue()
            task_count_queue.put(0)
            has_error = False

            with ThreadPoolExecutor(max_workers=WORKER_PROCESSES) as executor:
                futures = [executor.submit(self.process_single_video, input_file, selected_videos, output_format, total_tasks, task_count_queue) 
                           for input_file in self.input_files]
                for future in futures:
                    result = future.result()
                    if result is None:
                        has_error = True
                        break
                    else:
                        elapsed = time.time() - self.start_time
                        self.queue.put(("complete", result, elapsed))

            if not has_error and self.analytics_data:
                self.show_analytics_dashboard()
            logging.info("process_video_thread finished")
            log_session("Processing thread finished")
        except Exception as e:
            logging.error(f"Error in process_video_thread: {str(e)}", exc_info=True)
            log_session(f"Error in processing thread: {str(e)}")
            self.queue.put(("canceled", str(e)))

    def show_analytics_dashboard(self):
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

        log_session("Displayed analytics dashboard")

    def cancel_processing(self):
        global cancel_event
        cancel_event.set()
        logging.info("Cancel requested")
        log_session("Cancel processing requested")

    def process_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)

    def handle_message(self, message):
        msg_type, *args = message
        if msg_type == "task_start":
            task_name, progress = args
            self.current_task_label.configure(text=f"Current Task: {task_name}")
            self.progress.set(progress / 100)
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
        elif msg_type == "upload_progress":
            progress_value = args[0]
            self.progress.set(progress_value / 100)
            self.time_label.configure(text=f"Uploading: {progress_value:.2f}%")
            log_session(f"UI Update: Upload progress {progress_value:.2f}%")
        elif msg_type == "upload_status":
            status_text = args[0]
            self.current_task_label.configure(text=status_text)
            log_session(f"UI Update: Upload status: {status_text}")
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
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.switch_1h.configure(state="normal")
        self.custom_duration_entry.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.cancel_button.configure(state="disabled")
        log_session("UI reset to initial state")

    def get_youtube_client(self):
        if not hasattr(self, 'youtube_client'):
            logging.info("Initializing YouTube client")
            log_session("Initializing YouTube client")
            credentials = None
            if os.path.exists('token.pickle'):
                try:
                    with open('token.pickle', 'rb') as token:
                        credentials = pickle.load(token)
                    logging.info("Loaded credentials from token.pickle")
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
                        logging.info("Refreshed YouTube credentials")
                        log_session("Refreshed YouTube credentials")
                    except Exception as e:
                        logging.error(f"Failed to refresh credentials: {str(e)}")
                        log_session(f"Error: Failed to refresh credentials: {str(e)}")
                        credentials = None
                if not credentials:
                    if not os.path.exists('client_secrets.json'):
                        logging.error("client_secrets.json not found")
                        messagebox.showerror("Error", "client_secrets.json not found.")
                        log_session("Error: client_secrets.json not found for YouTube authentication")
                        return None
                    try:
                        logging.info("Starting OAuth flow")
                        log_session("Starting OAuth flow")
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'client_secrets.json',
                            scopes=['https://www.googleapis.com/auth/youtube.upload']
                        )
                        credentials = flow.run_local_server(port=0)
                        logging.info("OAuth authentication completed")
                        log_session("Authenticated with YouTube API")
                    except Exception as e:
                        logging.error(f"Failed to authenticate: {str(e)}")
                        log_session(f"Error: Failed to authenticate with YouTube API: {str(e)}")
                        messagebox.showerror("Error", f"Authentication failed: {str(e)}")
                        return None
            with open('token.pickle', 'wb') as token:
                pickle.dump(credentials, token)
                logging.info("Saved credentials to token.pickle")
                log_session("Saved YouTube credentials to token.pickle")
            http = httplib2.Http(timeout=120)
            authorized_http = google_auth_httplib2.AuthorizedHttp(credentials, http=http)
            self.youtube_client = build('youtube', 'v3', http=authorized_http)
            logging.info("YouTube client initialized successfully")
            log_session("YouTube client initialized")
        return self.youtube_client

    def start_upload(self, file_path, task_name, button):
        button.configure(state="disabled", text="Checking...")
        thread = threading.Thread(target=self.upload_to_youtube, args=(file_path, task_name, button))
        thread.start()
        logging.info(f"Started upload thread for {file_path}")
        log_session(f"Started YouTube upload for {file_path}")

    def upload_to_youtube(self, file_path, task_name, button):
        try:
            if not os.path.exists(file_path):
                self.root.after(0, lambda: messagebox.showerror("Error", f"Video file not found: {file_path}"))
                self.root.after(0, lambda b=button: b.configure(state="normal", text="Upload to YouTube"))
                log_session(f"Error: Video file not found for upload: {file_path}")
                return
            if not validate_video_file(file_path):
                self.root.after(0, lambda: messagebox.showerror("Error", "Invalid or corrupted video file."))
                self.root.after(0, lambda b=button: b.configure(state="normal", text="Upload to YouTube"))
                log_session(f"Error: Invalid or corrupted video file for upload: {file_path}")
                return
            if not check_network_stability():
                self.root.after(0, lambda: messagebox.showerror("Error", "Network unstable."))
                self.root.after(0, lambda b=button: b.configure(state="normal", text="Upload to YouTube"))
                log_session("Error: Network unstable, cannot upload to YouTube")
                return
            
            self.root.after(0, lambda b=button: b.configure(text="Uploading..."))
            
            youtube = self.get_youtube_client()
            if not youtube:
                logging.error("YouTube client is None, aborting upload")
                log_session("Error: Failed to get YouTube client")
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to authenticate with YouTube"))
                self.root.after(0, lambda b=button: b.configure(state="normal", text="Upload to YouTube"))
                return
            
            duration_str = task_name.split()[1]
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            title = file_name + (" #shorts" if duration_str == "60s" else "")
            description = "Uploaded via Bird Box Video Processor" + (" #shorts" if duration_str == "60s" else "")
            tags = ['bird', 'nature', 'video'] + (['#shorts'] if duration_str == "60s" else [])
            request_body = {
                'snippet': {'title': title, 'description': description, 'tags': tags, 'categoryId': '22'},
                'status': {'privacyStatus': 'unlisted'}
            }
            logging.info(f"Preparing to upload {file_path} with title: {title}")
            log_session(f"Uploading {file_path} with title: {title}")
            
            media = MediaFileUpload(file_path, resumable=True, chunksize=512 * 1024)
            request = youtube.videos().insert(part='snippet,status', body=request_body, media_body=media)
            
            self.queue.put(("upload_status", "Starting upload to YouTube..."))
            max_retries = 3
            initial_retry_delay = 5
            retry_delay = initial_retry_delay
            for attempt in range(max_retries):
                try:
                    response = None
                    while response is None:
                        status, response = request.next_chunk()
                        if status:
                            progress = status.progress() * 100
                            logging.info(f"Upload progress: {progress:.2f}%")
                            log_session(f"Upload progress: {progress:.2f}%")
                            self.queue.put(("upload_progress", progress))
                    self.queue.put(("upload_status", "Upload completed successfully"))
                    logging.info(f"Upload completed successfully for {file_path}")
                    log_session(f"Upload successful: {file_path}, YouTube URL: https://youtu.be/{response['id']}")
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Video uploaded: https://youtu.be/{response['id']}"))
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.queue.put(("upload_status", f"Upload failed, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"))
                        logging.warning(f"Upload attempt {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...")
                        log_session(f"Upload attempt {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        error_msg = str(e)
                        logging.error(f"Upload failed after {max_retries} attempts: {error_msg}")
                        log_session(f"Error: Upload failed after {max_retries} attempts: {error_msg}")
                        self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Upload failed: {msg}"))
        except Exception as e:
            error_message = str(e)
            logging.error(f"Unexpected error during upload: {error_message}", exc_info=True)
            log_session(f"Error: Unexpected error during upload: {error_message}")
            self.root.after(0, lambda msg=error_message: messagebox.showerror("Error", f"Unexpected error: {msg}"))
        finally:
            self.root.after(0, lambda b=button: b.configure(state="normal", text="Upload to YouTube"))

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