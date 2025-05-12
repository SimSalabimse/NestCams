import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import time
import threading
import queue
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Version number
VERSION = "2.1.2"  # Optimized for 24 FPS, 1920x1080

# Fixed video properties
FPS = 24
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

def compute_motion_score(prev_frame, current_frame, threshold=30, downscale_factor=0.5):
    """Compute motion score between downsampled frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_resized = cv2.resize(prev_frame, None, fx=downscale_factor, fy=downscale_factor)
    curr_resized = cv2.resize(current_frame, None, fx=downscale_factor, fy=downscale_factor)
    prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.sum(diff > threshold, dtype=np.uint32)

def is_white_or_black_frame(frame, white_threshold=200, black_threshold=50):
    """Check if frame is predominantly white or black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > white_threshold or mean_brightness < black_threshold

def normalize_frame(frame, clip_limit=1.0, saturation_multiplier=1.1):
    """Normalize frame brightness and enhance saturation."""
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
        return None

def detect_motion_segments(input_path, motion_threshold=3000, sample_interval=10, cancel_event=None, progress_callback=None):
    """Detect motion segments with progress updates for 24 FPS videos."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_segments = []
    in_motion = False
    segment_start = 0
    prev_frame = None
    
    for frame_idx in range(0, total_frames, sample_interval):
        if cancel_event and cancel_event.is_set():
            cap.release()
            return []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            motion_score = compute_motion_score(prev_frame, frame, downscale_factor=0.5)
            if motion_score > motion_threshold and not in_motion:
                in_motion = True
                segment_start = frame_idx
            elif motion_score <= motion_threshold and in_motion:
                in_motion = False
                motion_segments.append((segment_start, frame_idx))
        prev_frame = frame
        
        if progress_callback and frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 33
            progress_callback(progress, frame_idx, total_frames, 0)
    
    if in_motion:
        motion_segments.append((segment_start, total_frames - 1))
    
    cap.release()
    return motion_segments

def create_intermediate_video(input_path, motion_segments, output_path, cancel_event=None, progress_callback=None):
    """Create an intermediate video from motion segments for 24 FPS videos."""
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    total_motion_frames = sum(end - start + 1 for start, end in motion_segments)
    processed_frames = 0
    
    for start, end in motion_segments:
        if cancel_event and cancel_event.is_set():
            cap.release()
            out.release()
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            processed_frames += 1
            if progress_callback and processed_frames % 100 == 0:
                progress = 33 + (processed_frames / total_motion_frames) * 33
                progress_callback(progress, processed_frames, total_motion_frames, 0)
    
    cap.release()
    out.release()

def filter_and_adjust_speed(input_path, output_path, desired_duration, white_threshold=200, black_threshold=50, clip_limit=1.0, saturation_multiplier=1.1, progress_callback=None, cancel_event=None):
    """Filter unwanted frames and adjust speed for 24 FPS videos."""
    clip = VideoFileClip(input_path)
    filtered_frames = []
    total_frames = int(clip.duration * FPS)
    frame_count = 0
    
    for t in np.arange(0, clip.duration, 1 / FPS):
        if cancel_event and cancel_event.is_set():
            clip.close()
            return "Processing canceled by user"
        frame = clip.get_frame(t)
        if not is_white_or_black_frame(frame, white_threshold, black_threshold):
            filtered_frames.append(t)
        frame_count += 1
        if progress_callback and frame_count % 100 == 0:
            progress = 66 + (frame_count / total_frames) * 34
            progress_callback(progress, frame_count, total_frames, 0)
    
    if not filtered_frames:
        clip.close()
        return "No frames left after filtering"
    
    filtered_clip = concatenate_videoclips([clip.subclip(t, t + 1 / FPS) for t in filtered_frames])
    current_duration = filtered_clip.duration
    speed_factor = current_duration / desired_duration if current_duration > desired_duration else 1.0
    final_clip = filtered_clip.speedx(factor=speed_factor)
    
    def normalize(t):
        frame = final_clip.get_frame(t)
        normalized = normalize_frame(frame, clip_limit, saturation_multiplier)
        return normalized if normalized is not None else frame
    
    final_clip = final_clip.fl(normalize)
    final_clip.write_videofile(output_path, codec='libx264', fps=FPS)
    clip.close()
    return None

def process_video_multi_pass(input_path, output_path, desired_duration, motion_threshold, sample_interval, white_threshold, black_threshold, clip_limit, saturation_multiplier, progress_callback, cancel_event):
    """Execute multi-pass video processing with improved feedback for 24 FPS videos."""
    temp_path = f"temp_intermediate_{os.urandom(8).hex()}.mp4"
    
    motion_segments = detect_motion_segments(input_path, motion_threshold, sample_interval, cancel_event, progress_callback)
    if not motion_segments:
        return "No motion detected"
    
    create_intermediate_video(input_path, motion_segments, temp_path, cancel_event, progress_callback)
    if cancel_event and cancel_event.is_set():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return "Processing canceled by user"
    
    error = filter_and_adjust_speed(temp_path, output_path, desired_duration, white_threshold, black_threshold, clip_limit, saturation_multiplier, progress_callback, cancel_event)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if error:
        return error
    if progress_callback:
        progress_callback(100, 3, 3, 0)
    
    return None

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize GUI and app state for 24 FPS, 1920x1080 videos."""
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
        
        self.motion_label = ctk.CTkLabel(root, text="Motion Detection Sensitivity")
        self.motion_label.pack(pady=5)
        self.motion_slider = ctk.CTkSlider(root, from_=1000, to=10000, number_of_steps=90, command=self.update_motion_threshold)
        self.motion_slider.set(3000)
        self.motion_slider.pack(pady=5)
        self.motion_value_label = ctk.CTkLabel(root, text="Threshold: 3000")
        self.motion_value_label.pack(pady=5)
        
        self.brightness_label = ctk.CTkLabel(root, text="Brightness Thresholds")
        self.brightness_label.pack(pady=5)
        
        self.white_label = ctk.CTkLabel(root, text="White Threshold")
        self.white_label.pack(pady=2)
        self.white_slider = ctk.CTkSlider(root, from_=150, to=255, number_of_steps=105, command=self.update_white_threshold)
        self.white_slider.set(200)
        self.white_slider.pack(pady=2)
        self.white_value_label = ctk.CTkLabel(root, text="White: 200")
        self.white_value_label.pack(pady=2)
        
        self.black_label = ctk.CTkLabel(root, text="Black Threshold")
        self.black_label.pack(pady=2)
        self.black_slider = ctk.CTkSlider(root, from_=0, to=100, number_of_steps=100, command=self.update_black_threshold)
        self.black_slider.set(50)
        self.black_slider.pack(pady=2)
        self.black_value_label = ctk.CTkLabel(root, text="Black: 50")
        self.black_value_label.pack(pady=2)
        
        self.normalization_label = ctk.CTkLabel(root, text="Normalization Parameters")
        self.normalization_label.pack(pady=5)
        
        self.clip_label = ctk.CTkLabel(root, text="CLAHE Clip Limit")
        self.clip_label.pack(pady=2)
        self.clip_slider = ctk.CTkSlider(root, from_=0.5, to=5.0, number_of_steps=90, command=self.update_clip_limit)
        self.clip_slider.set(1.0)
        self.clip_slider.pack(pady=2)
        self.clip_value_label = ctk.CTkLabel(root, text="Clip Limit: 1.0")
        self.clip_value_label.pack(pady=2)
        
        self.saturation_label = ctk.CTkLabel(root, text="Saturation Multiplier")
        self.saturation_label.pack(pady=2)
        self.saturation_slider = ctk.CTkSlider(root, from_=0.5, to=2.0, number_of_steps=150, command=self.update_saturation_multiplier)
        self.saturation_slider.set(1.1)
        self.saturation_slider.pack(pady=2)
        self.saturation_value_label = ctk.CTkLabel(root, text="Saturation: 1.1")
        self.saturation_value_label.pack(pady=2)
        
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
        
        self.motion_threshold = 3000
        self.white_threshold = 200
        self.black_threshold = 50
        self.clip_limit = 1.0
        self.saturation_multiplier = 1.1
    
    def update_motion_threshold(self, value):
        self.motion_threshold = int(value)
        self.motion_value_label.configure(text=f"Threshold: {self.motion_threshold}")
    
    def update_white_threshold(self, value):
        self.white_threshold = int(value)
        self.white_value_label.configure(text=f"White: {self.white_threshold}")
    
    def update_black_threshold(self, value):
        self.black_threshold = int(value)
        self.black_value_label.configure(text=f"Black: {self.black_threshold}")
    
    def update_clip_limit(self, value):
        self.clip_limit = float(value)
        self.clip_value_label.configure(text=f"Clip Limit: {self.clip_limit:.1f}")
    
    def update_saturation_multiplier(self, value):
        self.saturation_multiplier = float(value)
        self.saturation_value_label.configure(text=f"Saturation: {self.saturation_multiplier:.1f}")
    
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
        """Process video with dynamic time estimation for 24 FPS videos."""
        selected_videos = []
        if self.generate_60s.get():
            selected_videos.append(("Generate 60s Video", 60))
        if self.generate_12min.get():
            selected_videos.append(("Generate 12min Video", 720))
        
        base, ext = os.path.splitext(self.input_file)
        output_files = {}
        
        cap = cv2.VideoCapture(self.input_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        estimated_total_time = (total_frames / 10) * 0.01  # Rough estimate: 0.01s per sampled frame
        
        for task_name, desired_duration in selected_videos:
            if self.cancel_event.is_set():
                self.queue.put(("canceled", "User Cancellation"))
                break
            output_file = f"{base}_{task_name.split()[1]}{ext}"
            self.queue.put(("task_start", task_name, 0))
            
            def progress_callback(progress, current, total, remaining):
                percentage = (current / total) * 100 if total > 0 else 0
                if remaining == 0:
                    remaining = estimated_total_time * (1 - current / total)
                self.queue.put(("progress", progress, percentage, remaining))
            
            error = process_video_multi_pass(
                self.input_file, output_file, desired_duration,
                motion_threshold=self.motion_threshold,
                sample_interval=10,
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
        """Stop processing."""
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

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()