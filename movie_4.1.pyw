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

# Version number
VERSION = "2.0.4"  # Updated for new features

def compute_motion_score(prev_frame, current_frame, threshold=30):
    """Compute motion score between two frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.sum(diff > threshold)

def is_white_or_black_frame(frame, white_threshold=200, black_threshold=50):
    """Check if frame is predominantly white or black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > white_threshold or mean_brightness < black_threshold

def normalize_frame(frame):
    """Normalize frame brightness and enhance saturation."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = (s.astype('float32') * 1.1).clip(0, 255).astype('uint8')
    hsv_enhanced = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

def process_video_multi_pass(input_path, output_path, desired_duration, motion_threshold=5000, sample_interval=10, progress_callback=None, cancel_event=None):
    """Process video in multiple passes: longer video, filter white/black, trim."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
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
            motion_score = compute_motion_score(prev_frame, frame)
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
        return "No frames selected in first pass"
    
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
        if not is_white_or_black_frame(frame):
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
            normalized_frame = normalize_frame(frame)
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

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize GUI and app state."""
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
            
            error = process_video_multi_pass(self.input_file, output_file, desired_duration, progress_callback=progress_callback, cancel_event=self.cancel_event)
            
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
            elapsed_min = elapsed / 60
            self.output_60s.configure(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_12min.configure(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Complete in {elapsed_min:.2f} minutes")
            self.reset_ui()
        elif message[0] == "canceled":
            reason = message[1]
            self.output_60s.configure(text=f"60s Video: Canceled - {reason}")
            self.output_12min.configure(text=f"12min Video: Canceled - {reason}")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text="Process Canceled")
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