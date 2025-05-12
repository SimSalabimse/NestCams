import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import time
import threading
import queue

# Version number
VERSION = "2.0.0"

def compute_motion_score(prev_frame, current_frame, threshold=30):
    """Compute motion score between two frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.sum(diff > threshold)

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

def process_video_two_pass(input_path, output_path, desired_duration, motion_threshold=5000, brightness_threshold=200):
    """Process video in two passes: motion detection then frame writing."""
    # First Pass: Motion Detection and Frame Selection
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
    
    # Determine frames to sample based on desired duration
    target_frames = int(desired_duration * fps)
    sample_interval = max(1, total_frames // (target_frames * 2))  # Oversample slightly
    
    selected_indices = []
    prev_frame = None
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            motion_score = compute_motion_score(prev_frame, frame)
            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            if motion_score > motion_threshold and brightness <= brightness_threshold:
                selected_indices.append(frame_idx)
        
        prev_frame = frame
        frame_idx += 1
    
    cap.release()
    
    if not selected_indices:
        return "No frames selected after filtering"
    
    # Adjust selected frames to fit desired duration
    if len(selected_indices) > target_frames:
        step = len(selected_indices) / target_frames
        selected_indices = [selected_indices[int(i * step)] for i in range(target_frames)]
    
    # Second Pass: Process and Write Selected Frames
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    selected_idx = 0
    
    while cap.isOpened() and selected_idx < len(selected_indices):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx == selected_indices[selected_idx]:
            normalized_frame = normalize_frame(frame)
            out.write(normalized_frame)
            selected_idx += 1
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return None if selected_idx > 0 else "No frames written"

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
            self.worker_thread = threading.Thread(target=self.process_video_thread)
            self.worker_thread.start()
    
    def process_video_thread(self):
        """Process video in a background thread."""
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
            
            start_time = time.time()
            error = process_video_two_pass(self.input_file, output_file, desired_duration)
            
            if error:
                self.queue.put(("canceled", error))
                break
            else:
                output_files[task_name] = output_file
                elapsed = time.time() - start_time
                self.queue.put(("progress", 100, int(elapsed // 60)))
        
        if not self.cancel_event.is_set():
            self.queue.put(("complete", output_files.get("Generate 60s Video"), output_files.get("Generate 12min Video")))
    
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
            self.time_label.configure(text="Processing...")
        elif message[0] == "progress":
            progress_value, elapsed_time = message[1:]
            self.progress.set(progress_value / 100)
            self.time_label.configure(text=f"Elapsed Time: {elapsed_time} minutes")
        elif message[0] == "complete":
            output_60s, output_12min = message[1:]
            self.output_60s.configure(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_12min.configure(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text="Process Complete")
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