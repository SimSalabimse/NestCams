import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, VideoClip
import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import bisect

def find_motion_segments(motion_scores, threshold=5000, min_segment_length=3, merge_gap=2):
    """
    Identify continuous segments with motion above the threshold.
    """
    segments = []
    start = None
    duration = len(motion_scores)
    
    for t, score in enumerate(motion_scores):
        if score > threshold:
            if start is None:
                start = t
        else:
            if start is not None:
                end = t
                if end - start >= min_segment_length:
                    if segments and start - segments[-1][1] <= merge_gap:
                        segments[-1] = (segments[-1][0], end)
                    else:
                        segments.append((start, end))
                start = None
    
    if start is not None:
        end = duration
        if end - start >= min_segment_length:
            if segments and start - segments[-1][1] <= merge_gap:
                segments[-1] = (segments[-1][0], end)
            else:
                segments.append((start, end))
    
    return segments

def compute_frame_motion(t, video_path, threshold=30):
    """
    Compute motion score for a specific second in the video.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_MSEC, (t - 1) * 1000)
    ret, prev_frame = cap.read()
    cap.release()
    if not ret or prev_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    return np.sum(diff > threshold)

def normalize_frame(frame):
    """
    Normalize brightness of a single frame using CLAHE and enhance color saturation with optimized memory usage.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Use float32 to reduce memory usage from float64
    s = (s.astype('float32') * 1.1).clip(0, 255).astype('uint8')
    hsv_enhanced = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

def compute_brightness(t, clip):
    """Compute the brightness of a frame at time t."""
    frame = clip.get_frame(t)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def process_clip_frames(clip, fps, progress_callback, cancel_event):
    """Process clip frames with enhanced brightness filtering and normalization, showing progress."""
    try:
        file_size = os.path.getsize(clip.filename) / (1024 * 1024)  # in MB
        if file_size > 1000:
            max_workers = max(1, os.cpu_count() // 2)
        else:
            max_workers = os.cpu_count()
        
        duration = clip.duration
        frame_times = [t for t in np.arange(0, duration, 1 / fps)]
        total_frames = len(frame_times)
        
        # Step 1: Compute brightness for all frames in parallel
        brightnesses = [None] * total_frames
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_brightness, t, clip): i for i, t in enumerate(frame_times)}
            for future in as_completed(futures):
                if cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    return (None, "User Cancellation")
                i = futures[future]
                brightnesses[i] = future.result()
        
        # Step 2: Decide which frames to keep based on brightness history with absolute threshold
        kept_times = []
        brightness_history = []
        for t, brightness in zip(frame_times, brightnesses):
            if len(brightness_history) > 1:
                median_brightness = np.median(brightness_history)
                # Enhanced filtering: relative thresholds plus absolute upper limit for white frames
                if brightness < 0.3 * median_brightness or brightness > 1.7 * median_brightness or brightness > 200:
                    continue
            kept_times.append(t)
            if len(brightness_history) >= 5:
                brightness_history.pop(0)
            brightness_history.append(brightness)
        
        if not kept_times:
            return (None, "No frames kept after filtering")
        
        # Step 3: Normalize the kept frames in parallel
        processed_frames = [None] * len(kept_times)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(normalize_frame, clip.get_frame(t)): i for i, t in enumerate(kept_times)}
            for future in as_completed(futures):
                if cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    return (None, "User Cancellation")
                i = futures[future]
                processed_frames[i] = future.result()
                progress_callback(i + 1, len(kept_times))
        
        # Step 4: Create the VideoClip
        def make_frame(t):
            if not kept_times:
                return np.zeros((clip.h, clip.w, 3), dtype='uint8')
            idx = bisect.bisect_right(kept_times, t)
            if idx == 0:
                return processed_frames[0]
            else:
                return processed_frames[idx - 1]
        
        return (VideoClip(make_frame, duration=duration).set_fps(fps), None)
    except MemoryError:
        return (None, "Memory Error: Insufficient memory")
    except Exception as e:
        return (None, f"Error: {str(e)}")

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize GUI and app state."""
        self.version = "1.0.1"
        self.root = root
        self.root.title(f"Bird Box Video Processor v{self.version}")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.version_label = ctk.CTkLabel(root, text=f"Version: {self.version}")
        self.version_label.pack(pady=5)
        
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
        
        total_tasks = 1 + len(selected_videos)
        task_share = 100 / total_tasks
        
        self.queue.put(("task_start", "Motion Detection", 0))
        motion_scores = self.compute_motion_scores_with_progress(self.input_file, task_share, 0)
        if motion_scores is None:
            self.queue.put(("canceled", "Failed to compute motion scores"))
            return
        elif self.cancel_event.is_set():
            self.queue.put(("canceled", "User Cancellation"))
            return
        
        segments = find_motion_segments(motion_scores)
        if not segments:
            self.queue.put(("complete", "No motion detected", "No motion detected"))
            return
        
        base, ext = os.path.splitext(self.input_file)
        output_files = {}
        video = VideoFileClip(self.input_file)
        fps = video.fps
        
        try:
            adjusted_clips = []
            for segment_index, (start, end) in enumerate(segments):
                if self.cancel_event.is_set():
                    self.queue.put(("canceled", "User Cancellation"))
                    break
                clip = video.subclip(start, end)
                self.queue.put(("task_start", f"Processing Segment {segment_index + 1}", task_share))
                
                def progress_callback(processed, total):
                    if self.cancel_event.is_set():
                        return
                    progress = (processed / total) * task_share
                    self.queue.put(("progress", progress, 0))
                
                result = process_clip_frames(clip, fps, progress_callback, self.cancel_event)
                if isinstance(result, tuple):
                    normalized_clip, error = result
                    if error:
                        self.queue.put(("canceled", error))
                        break
                    elif normalized_clip is None:
                        self.queue.put(("canceled", "User Cancellation"))
                        break
                    else:
                        adjusted_clips.append(normalized_clip.fx(vfx.fadein, 1.0).fx(vfx.fadeout, 1.0))
                self.queue.put(("task_complete", task_share * (segment_index + 1)))
            
            if not self.cancel_event.is_set():
                for i, (task_name, desired_duration) in enumerate(selected_videos, start=1):
                    if self.cancel_event.is_set():
                        self.queue.put(("canceled", "User Cancellation"))
                        break
                    start_progress = (1 + i) * task_share
                    self.queue.put(("task_start", task_name, start_progress))
                    
                    total_duration = sum(end - start for start, end in segments)
                    if total_duration > 0:
                        speed_factor = total_duration / desired_duration
                        concatenated = concatenate_videoclips(adjusted_clips)
                        sped_up = concatenated.speedx(speed_factor)
                        output_file = f"{base}_{task_name.split()[1]}{ext}"
                        
                        if os.path.exists(output_file):
                            confirm = messagebox.askokcancel("File Exists", f"File {output_file} already exists. Overwrite?")
                            if not confirm:
                                self.queue.put(("canceled", "User chose not to overwrite"))
                                continue
                        
                        sped_up.write_videofile(output_file, verbose=False, logger=None)
                        output_files[task_name] = output_file
                    
                    self.queue.put(("task_complete", start_progress + task_share))
                
                if not self.cancel_event.is_set():
                    self.queue.put(("complete", output_files.get("Generate 60s Video"), output_files.get("Generate 12min Video")))
        except Exception as e:
            self.queue.put(("canceled", f"Error: {str(e)}"))
        finally:
            video.close()
    
    def compute_motion_scores_with_progress(self, video_path, task_share, start_progress, threshold=30):
        """
        Compute motion scores with progress updates using parallel processing.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_seconds = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        cap.release()
        
        if total_seconds <= 0:
            return None
        
        motion_scores = [0] * total_seconds
        if total_seconds > 1:
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # in MB
            if file_size > 1000:
                max_workers = max(1, os.cpu_count() // 2)
            else:
                max_workers = os.cpu_count()
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(compute_frame_motion, t, video_path, threshold): t for t in range(1, total_seconds)}
                completed_count = 0
                for future in as_completed(futures):
                    if self.cancel_event.is_set():
                        for f in futures:
                            f.cancel()
                        return None
                    t = futures[future]
                    motion_scores[t] = future.result()
                    completed_count += 1
                    progress = start_progress + (completed_count / (total_seconds - 1)) * task_share
                    elapsed_time = time.time() - start_time
                    if completed_count > 0:
                        time_per_task = elapsed_time / completed_count
                        remaining_tasks = (total_seconds - 1) - completed_count
                        remaining_time = remaining_tasks * time_per_task
                        self.queue.put(("progress", progress, int(remaining_time // 60)))
                    else:
                        self.queue.put(("progress", progress, 0))
        return motion_scores
    
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
        if message[0] == "progress":
            progress_value, remaining_time = message[1:]
            self.progress.set(progress_value / 100)
            self.time_label.configure(text=f"Estimated Time Remaining: {remaining_time} minutes")
        elif message[0] == "task_start":
            task_name, start_progress = message[1:]
            self.current_task_label.configure(text=f"Current Task: {task_name}")
            self.progress.set(start_progress / 100)
            if task_name != "Motion Detection":
                self.time_label.configure(text="Processing...")
        elif message[0] == "task_complete":
            progress_value = message[1]
            self.progress.set(progress_value / 100)
        elif message[0] == "complete":
            output_60s, output_12min = message[1:]
            self.output_60s.configure(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_12min.configure(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text="Process Complete")
            self.reset_ui()
        elif message[0] == "canceled":
            reason = message[1] if len(message) > 1 else "Unknown reason"
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