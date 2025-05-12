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
VERSION = "2.0.3"  # Updated for new features

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

def process_video_two_pass(input_path, output_path, desired_duration, motion_threshold=5000, sample_interval=10, buffer_size=5, progress_callback=None, cancel_event=None):
    """Process video in two passes with progress updates and cancellation support."""
    # First Pass: Motion Detection and Flash Removal
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
    
    in_motion = False
    current_segment_brightnesses = []
    include_indices = []
    exclude_indices = set()
    prev_frame = None
    
    print(f"Starting first pass: {total_frames} frames to process")
    for frame_idx in range(total_frames):
        if cancel_event and cancel_event.is_set():
            cap.release()
            return "Processing canceled by user"
        ret, frame = cap.read()
        if not ret:
            print(f"First pass stopped at frame {frame_idx}: failed to read frame")
            break
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if frame_idx % sample_interval == 0:
            if prev_frame is not None:
                motion_score = compute_motion_score(prev_frame, frame)
                if motion_score > motion_threshold and not in_motion:
                    in_motion = True
                    current_segment_brightnesses = [brightness]
                    include_indices.append(frame_idx)
                elif motion_score <= motion_threshold and in_motion:
                    in_motion = False
                    if current_segment_brightnesses:
                        median_brightness = np.median(current_segment_brightnesses)
                        for i in range(min(buffer_size, len(current_segment_brightnesses))):
                            if abs(current_segment_brightnesses[i] - median_brightness) / median_brightness > 0.2:
                                exclude_indices.add(include_indices[-len(current_segment_brightnesses) + i])
                        for i in range(max(0, len(current_segment_brightnesses) - buffer_size), len(current_segment_brightnesses)):
                            if abs(current_segment_brightnesses[i] - median_brightness) / median_brightness > 0.2:
                                exclude_indices.add(include_indices[-len(current_segment_brightnesses) + i])
                    current_segment_brightnesses = []
                else:
                    if in_motion:
                        include_indices.append(frame_idx)
                        current_segment_brightnesses.append(brightness)
            prev_frame = frame
        else:
            if in_motion:
                include_indices.append(frame_idx)
                current_segment_brightnesses.append(brightness)
        
        # Progress update for first pass (0-50%)
        if progress_callback and frame_idx % 100 == 0:
            progress_callback(frame_idx / total_frames * 50, frame_idx, total_frames)
    
    if in_motion and current_segment_brightnesses:
        median_brightness = np.median(current_segment_brightnesses)
        for i in range(min(buffer_size, len(current_segment_brightnesses))):
            if abs(current_segment_brightnesses[i] - median_brightness) / median_brightness > 0.2:
                exclude_indices.add(include_indices[-len(current_segment_brightnesses) + i])
        for i in range(max(0, len(current_segment_brightnesses) - buffer_size), len(current_segment_brightnesses)):
            if abs(current_segment_brightnesses[i] - median_brightness) / median_brightness > 0.2:
                exclude_indices.add(include_indices[-len(current_segment_brightnesses) + i])
    
    cap.release()
    print(f"First pass complete: {len(include_indices)} frames selected initially")
    
    # Filter include_indices
    final_include_indices = [idx for idx in include_indices if idx not in exclude_indices]
    print(f"After filtering: {len(final_include_indices)} frames remain")
    
    if not final_include_indices:
        return "No frames selected after filtering"
    
    # Adjust to desired duration
    target_frames = int(desired_duration * fps)
    if len(final_include_indices) > target_frames:
        step = len(final_include_indices) / target_frames
        final_include_indices = [final_include_indices[int(i * step)] for i in range(target_frames)]
    print(f"Adjusted to {len(final_include_indices)} frames for {desired_duration}s")
    
    # Second Pass: Write selected frames
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print("Starting second pass")
    for idx, frame_idx in enumerate(final_include_indices):
        if cancel_event and cancel_event.is_set():
            cap.release()
            out.release()
            return "Processing canceled by user"
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            normalized_frame = normalize_frame(frame)
            out.write(normalized_frame)
        else:
            print(f"Second pass warning: failed to read frame {frame_idx}")
        
        # Progress update for second pass (50-100%)
        if progress_callback and idx % 10 == 0:
            progress_callback(50 + (idx / len(final_include_indices) * 50), idx, len(final_include_indices))
    
    cap.release()
    out.release()
    print("Second pass complete")
    
    return None if len(final_include_indices) > 0 else "No frames written"

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
            
            def progress_callback(progress, current_frame, total_frames):
                percentage = (current_frame / total_frames) * 100 if total_frames > 0 else 0
                self.queue.put(("progress", progress, percentage))
            
            error = process_video_two_pass(self.input_file, output_file, desired_duration, progress_callback=progress_callback, cancel_event=self.cancel_event)
            
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
            self.time_label.configure(text="Processing...")
        elif message[0] == "progress":
            progress_value, percentage = message[1:]
            self.progress.set(progress_value / 100)
            self.time_label.configure(text=f"Processing: {percentage:.2f}% complete")
        elif message[0] == "complete":
            output_60s, output_12min, elapsed = message[1:]
            self.output_60s.configure(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_12min.configure(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Complete in {elapsed:.2f} seconds")
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