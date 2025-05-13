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

# Version number
VERSION = "2.0.6"  # Updated to reflect settings page addition

### Helper Functions

def compute_motion_score(prev_frame, current_frame, threshold=30):
    """Compute motion score between two frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.sum(diff > threshold, dtype=np.uint32)  # Avoid float64 accumulation

def is_white_or_black_frame(frame, white_threshold=200, black_threshold=50):
    """Check if frame is predominantly white or black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > white_threshold or mean_brightness < black_threshold

def normalize_frame(frame, clip_limit=1.0, saturation_multiplier=1.1):
    """Normalize frame brightness and enhance saturation with optimized memory usage."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = (s.astype('float32') * saturation_multiplier).clip(0, 255).astype('uint8')
        hsv_enhanced = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    except MemoryError:
        return None  # Handle memory error gracefully

def process_video_multi_pass(input_path, output_path, desired_duration, motion_threshold=3000, sample_interval=5, white_threshold=200, black_threshold=50, clip_limit=1.0, saturation_multiplier=1.1, progress_callback=None, cancel_event=None):
    """Process video in multiple passes: longer video, filter white/black, trim."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return "Failed to open video file"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0 or fps <= 0:
        cap.release()
        return "Invalid video properties"
    
    # Step 1: Generate longer intermediate video (e.g., 3x desired duration)
    intermediate_frames = int(desired_duration * fps * 3)  # 3x longer initially
    temp_path1 = f"temp_intermediate_{uuid.uuid4().hex}.mp4"
    
    in_motion = False
    include_indices = []
    prev_frame = None
    start_time = time.time()
    
    print(f"Pass 1: Selecting {intermediate_frames} frames with motion")
    for frame_idx in range(total_frames):
        if cancel_event and cancel_event.is_set():
            cap.release()
            return "Processing canceled by user"
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0 and prev_frame is not None:
            motion_score = compute_motion_score(prev_frame, frame, threshold=motion_threshold)
            if motion_score > motion_threshold:
                in_motion = True
                include_indices.append(frame_idx)
            elif motion_score <= motion_threshold and in_motion:
                in_motion = False
            prev_frame = frame
        elif in_motion:
            include_indices.append(frame_idx)
        
        if len(include_indices) >= intermediate_frames:
            break
        
        # Progress for Pass 1 (0-33%)
        elapsed = time.time() - start_time
        if progress_callback and frame_idx % 100 == 0:
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
            progress_callback(frame_idx / total_frames * 33, frame_idx, total_frames, remaining)
    
    cap.release()
    if not include_indices:
        # Fallback: Include evenly spaced frames if no motion detected
        include_indices = [i for i in range(0, min(total_frames, intermediate_frames), max(1, total_frames // intermediate_frames))]
        print("Warning: No motion detected, using fallback frames")
    
    # Write intermediate video
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path1, fourcc, fps, (frame_width, frame_height))
    for idx, frame_idx in enumerate(include_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        if progress_callback and idx % 50 == 0:
            progress_callback(33 + (idx / len(include_indices) * 33), idx, len(include_indices), 0)
    cap.release()
    out.release()
    
    # Step 2: Filter white/black frames
    temp_path2 = f"temp_filtered_{uuid.uuid4().hex}.mp4"
    cap = cv2.VideoCapture(temp_path1)
    out = cv2.VideoWriter(temp_path2, fourcc, fps, (frame_width, frame_height))
    filtered_indices = []
    
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if cancel_event and cancel_event.is_set():
            cap.release()
            out.release()
            os.remove(temp_path1)
            return "Processing canceled by user"
        ret, frame = cap.read()
        if not ret:
            break
        if not is_white_or_black_frame(frame, white_threshold, black_threshold):
            out.write(frame)
            filtered_indices.append(frame_idx)
        
        # Progress for Pass 2 (33-66%)
        if progress_callback and frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (cap.get(cv2.CAP_PROP_FRAME_COUNT) - frame_idx) / rate if rate > 0 else 0
            progress_callback(33 + (frame_idx / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 33), frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), remaining)
    
    cap.release()
    out.release()
    os.remove(temp_path1)
    
    # Step 3: Trim to final length
    target_frames = int(desired_duration * fps)
    cap = cv2.VideoCapture(temp_path2)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if len(filtered_indices) > target_frames:
        step = len(filtered_indices) / target_frames
        final_indices = [filtered_indices[int(i * step)] for i in range(target_frames)]
    else:
        final_indices = filtered_indices
    
    for idx, frame_idx in enumerate(final_indices):
        if cancel_event and cancel_event.is_set():
            cap.release()
            out.release()
            os.remove(temp_path2)
            return "Processing canceled by user"
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            normalized_frame = normalize_frame(frame, clip_limit, saturation_multiplier)
            if normalized_frame is None:
                print("Memory error during frame normalization")
                cap.release()
                out.release()
                os.remove(temp_path2)
                return "Memory error during processing"
            out.write(normalized_frame)
        
        # Progress for Pass 3 (66-100%)
        if progress_callback and idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(final_indices) - idx) / rate if rate > 0 else 0
            progress_callback(66 + (idx / len(final_indices) * 34), idx, len(final_indices), remaining)
    
    cap.release()
    out.release()
    os.remove(temp_path2)
    return None if final_indices else "No frames written after trimming"

### Main Application Class

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize GUI and app state with adjustable settings."""
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.label = ctk.CTkLabel(root, text="Select Input Video")
        self.label.pack(pady=10)
        
        self.generate_60s = tk.BooleanVar(value=True)
        self.switch_60s = ctk.CTkSwitch(root, text="Generate 60s Video", variable=self.generate_60s)
        self.switch_60s.pack(pady=5)
        
        self.generate_12min = tk.BooleanVar(value=True)
        self.switch_12min = ctk.CTkSwitch(root, text="Generate 12min Video", variable=self.generate_12min)
        self.switch_12min.pack(pady=5)
        
        self.settings_button = ctk.CTkButton(root, text="Settings", command=self.open_settings)
        self.settings_button.pack(pady=5)
        
        self.browse_button = ctk.CTkButton(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)
        
        self.progress = ctk.CTkProgressBar(root, orientation="horizontal", mode="determinate", width=300)
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
        
        self.output_60s = ctk.CTkLabel(root, text="60s Video: N/A")
        self.output_60s.pack(pady=5)
        
        self.output_12min = ctk.CTkLabel(root, text="12min Video: N/A")
        self.output_12min.pack(pady=5)
        
        self.input_file = None
        self.cancel_event = threading.Event()
        self.queue = queue.Queue()
        self.start_time = None
        self.root.after(100, self.process_queue)
        
        # Default settings
        self.motion_threshold = 3000
        self.white_threshold = 200
        self.black_threshold = 50
        self.clip_limit = 1.0
        self.saturation_multiplier = 1.1
        
        # Load settings from file if exists
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.motion_threshold = int(settings.get("motion_threshold", 3000))
                self.white_threshold = int(settings.get("white_threshold", 200))
                self.black_threshold = int(settings.get("black_threshold", 50))
                self.clip_limit = float(settings.get("clip_limit", 1.0))
                self.saturation_multiplier = float(settings.get("saturation_multiplier", 1.1))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            print("Warning: Could not load settings, using defaults")
    
    def open_settings(self):
        """Open the settings window with sliders for adjustable parameters."""
        self.settings_window = ctk.CTkToplevel(self.root)
        self.settings_window.title("Settings")
        
        # Motion Detection Sensitivity
        motion_label = ctk.CTkLabel(self.settings_window, text="Motion Detection Sensitivity")
        motion_label.pack(pady=5)
        self.motion_slider_settings = ctk.CTkSlider(self.settings_window, from_=1000, to=10000, number_of_steps=90)
        self.motion_slider_settings.set(self.motion_threshold)
        self.motion_slider_settings.pack(pady=5)
        motion_value_label = ctk.CTkLabel(self.settings_window, text=f"Threshold: {self.motion_threshold}")
        motion_value_label.pack(pady=5)
        def update_motion_label(value):
            motion_value_label.configure(text=f"Threshold: {int(float(value))}")
        self.motion_slider_settings.configure(command=update_motion_label)
        
        # White Threshold
        white_label = ctk.CTkLabel(self.settings_window, text="White Threshold")
        white_label.pack(pady=2)
        self.white_slider_settings = ctk.CTkSlider(self.settings_window, from_=150, to=255, number_of_steps=105)
        self.white_slider_settings.set(self.white_threshold)
        self.white_slider_settings.pack(pady=2)
        white_value_label = ctk.CTkLabel(self.settings_window, text=f"White: {self.white_threshold}")
        white_value_label.pack(pady=2)
        def update_white_label(value):
            white_value_label.configure(text=f"White: {int(float(value))}")
        self.white_slider_settings.configure(command=update_white_label)
        
        # Black Threshold
        black_label = ctk.CTkLabel(self.settings_window, text="Black Threshold")
        black_label.pack(pady=2)
        self.black_slider_settings = ctk.CTkSlider(self.settings_window, from_=0, to=100, number_of_steps=100)
        self.black_slider_settings.set(self.black_threshold)
        self.black_slider_settings.pack(pady=2)
        black_value_label = ctk.CTkLabel(self.settings_window, text=f"Black: {self.black_threshold}")
        black_value_label.pack(pady=2)
        def update_black_label(value):
            black_value_label.configure(text=f"Black: {int(float(value))}")
        self.black_slider_settings.configure(command=update_black_label)
        
        # CLAHE Clip Limit
        clip_label = ctk.CTkLabel(self.settings_window, text="CLAHE Clip Limit")
        clip_label.pack(pady=2)
        self.clip_slider_settings = ctk.CTkSlider(self.settings_window, from_=0.5, to=5.0, number_of_steps=90)
        self.clip_slider_settings.set(self.clip_limit)
        self.clip_slider_settings.pack(pady=2)
        clip_value_label = ctk.CTkLabel(self.settings_window, text=f"Clip Limit: {self.clip_limit:.1f}")
        clip_value_label.pack(pady=2)
        def update_clip_label(value):
            clip_value_label.configure(text=f"Clip Limit: {float(value):.1f}")
        self.clip_slider_settings.configure(command=update_clip_label)
        
        # Saturation Multiplier
        saturation_label = ctk.CTkLabel(self.settings_window, text="Saturation Multiplier")
        saturation_label.pack(pady=2)
        self.saturation_slider_settings = ctk.CTkSlider(self.settings_window, from_=0.5, to=2.0, number_of_steps=150)
        self.saturation_slider_settings.set(self.saturation_multiplier)
        self.saturation_slider_settings.pack(pady=2)
        saturation_value_label = ctk.CTkLabel(self.settings_window, text=f"Saturation: {self.saturation_multiplier:.1f}")
        saturation_value_label.pack(pady=2)
        def update_saturation_label(value):
            saturation_value_label.configure(text=f"Saturation: {float(value):.1f}")
        self.saturation_slider_settings.configure(command=update_saturation_label)
        
        # Save and Reset buttons
        save_button = ctk.CTkButton(self.settings_window, text="Save", command=self.save_settings)
        save_button.pack(pady=10)
        reset_button = ctk.CTkButton(self.settings_window, text="Reset to Default", command=self.reset_to_default)
        reset_button.pack(pady=10)
    
    def save_settings(self):
        """Save the current settings from the sliders to the instance variables and to settings.json."""
        self.motion_threshold = int(self.motion_slider_settings.get())
        self.white_threshold = int(self.white_slider_settings.get())
        self.black_threshold = int(self.black_slider_settings.get())
        self.clip_limit = float(self.clip_slider_settings.get())
        self.saturation_multiplier = float(self.saturation_slider_settings.get())
        
        settings = {
            "motion_threshold": self.motion_threshold,
            "white_threshold": self.white_threshold,
            "black_threshold": self.black_threshold,
            "clip_limit": self.clip_limit,
            "saturation_multiplier": self.saturation_multiplier
        }
        
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        
        self.settings_window.destroy()
    
    def reset_to_default(self):
        """Reset sliders to default values and save the settings."""
        self.motion_slider_settings.set(3000)
        self.white_slider_settings.set(200)
        self.black_slider_settings.set(50)
        self.clip_slider_settings.set(1.0)
        self.saturation_slider_settings.set(1.1)
        self.save_settings()
    
    def browse_file(self):
        """Handle file selection and start processing."""
        self.input_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.input_file:
            self.label.configure(text=f"Selected: {os.path.basename(self.input_file)}")
            if not self.generate_60s.get() and not self.generate_12min.get():
                messagebox.showwarning("Warning", "Please enable at least one video to generate.")
                return
            self.output_60s.configure(text="60s Video: Processing")
            self.output_12min.configure(text="12min Video: Processing")
            self.switch_60s.configure(state="disabled")
            self.switch_12min.configure(state="disabled")
            self.browse_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
            self.cancel_event.clear()
            self.start_time = time.time()
            self.worker_thread = threading.Thread(target=self.process_video_thread)
            self.worker_thread.start()
    
    def process_video_thread(self):
        """Process video in a background thread with progress updates."""
        selected_videos = []
        if self.generate_60s.get():
            selected_videos.append(("Generate 60s Video", 60))
        if self.generate_12min.get():
            selected_videos.append(("Generate 12min Video", 720))
        
        base, ext = os.path.splitext(self.input_file)
        output_files = {}
        
        for task_name, desired_duration in selected_videos:
            if self.cancel_event.is_set():
                self.queue.put(("canceled", "User Cancellation"))
                break
            output_file = f"{base}_{task_name.split()[1]}{ext}"
            self.queue.put(("task_start", task_name, 0))
            
            def progress_callback(progress, current, total, remaining):
                percentage = (current / total) * 100 if total > 0 else 0
                self.queue.put(("progress", progress, percentage, remaining))
            
            error = process_video_multi_pass(
                self.input_file, output_file, desired_duration,
                motion_threshold=self.motion_threshold,
                sample_interval=5,
                white_threshold=self.white_threshold,
                black_threshold=self.black_threshold,
                clip_limit=self.clip_limit,
                saturation_multiplier=self.saturation_multiplier,
                progress_callback=progress_callback,
                cancel_event=self.cancel_event
            )
            
            if error:
                self.queue.put(("canceled", error))
                break
            else:
                output_files[task_name] = output_file
                elapsed = time.time() - self.start_time
                self.queue.put(("complete", output_files.get("Generate 60s Video"), output_files.get("Generate 12min Video"), elapsed))
    
    def cancel_processing(self):
        """Stop processing by setting cancel event."""
        self.cancel_event.set()
    
    def process_queue(self):
        """Update GUI from queue messages."""
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)
    
    def handle_message(self, message):
        """Handle queue messages to update UI."""
        if message[0] == "task_start":
            task_name, progress = message[1:]
            self.current_task_label.configure(text=f"Current Task: {task_name}")
            self.progress.set(progress / 100)
            self.time_label.configure(text="Estimating time...")
        elif message[0] == "progress":
            progress_value, percentage, remaining = message[1:]
            self.progress.set(progress_value / 100)
            remaining_min = remaining / 60 if remaining > 0 else 0
            self.time_label.configure(text=f"Est. Time Remaining: {remaining_min:.2f} min ({percentage:.2f}%)")
        elif message[0] == "complete":
            output_60s, output_12min, elapsed = message[1:]
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes} min {seconds} sec"
            self.output_60s.configure(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_12min.configure(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Complete in {time_str}")
            self.reset_ui()
        elif message[0] == "canceled":
            reason = message[1]
            elapsed = time.time() - self.start_time if self.start_time else 0
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes} min {seconds} sec"
            self.output_60s.configure(text=f"60s Video: Canceled - {reason}")
            self.output_12min.configure(text=f"12min Video: Canceled - {reason}")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Canceled in {time_str}")
            self.reset_ui()
    
    def reset_ui(self):
        """Reset UI to initial state."""
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

### Entry Point

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()