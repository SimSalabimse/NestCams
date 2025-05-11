import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import os
import time
import threading
import queue

def find_motion_segments(motion_scores, threshold=10000, min_segment_length=5, merge_gap=1):
    """Identify all continuous segments with motion above the threshold."""
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

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Box Video Processor")
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
        self.input_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.input_file:
            self.label.configure(text=f"Selected: {os.path.basename(self.input_file)}")
            if not self.generate_60s.get() and not self.generate_12min.get():
                messagebox.showwarning("Warning", "Please enable at least one video to generate.")
                return
            self.switch_60s.configure(state="disabled")
            self.switch_12min.configure(state="disabled")
            self.browse_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
            self.cancel_event.clear()
            self.worker_thread = threading.Thread(target=self.process_video_thread)
            self.worker_thread.start()
    
    def process_video_thread(self):
        selected_videos = []
        if self.generate_60s.get():
            selected_videos.append(("Generate 60s Video", 60))
        if self.generate_12min.get():
            selected_videos.append(("Generate 12min Video", 720))
        
        total_tasks = 1 + len(selected_videos)  # 1 for motion detection
        task_share = 100 / total_tasks
        
        # Motion Detection
        self.queue.put(("task_start", "Motion Detection", 0))
        motion_scores = self.compute_motion_scores_with_progress(self.input_file, task_share, 0)
        if motion_scores is None or self.cancel_event.is_set():
            self.queue.put(("canceled",))
            return
        
        segments = find_motion_segments(motion_scores)
        if not segments:
            self.queue.put(("complete", "No motion detected", "No motion detected"))
            return
        
        base, ext = os.path.splitext(self.input_file)
        output_files = {}
        video = VideoFileClip(self.input_file)
        
        try:
            # Extract clips and calculate average brightness
            clips = []
            brightness_levels = []
            cap = cv2.VideoCapture(self.input_file)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            for start, end in segments:
                clip = video.subclip(start, end)
                segment_brightness = []
                for t in range(start, end, max(1, (end - start) // 5)):
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000 / fps)
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        segment_brightness.append(np.mean(gray))
                avg_brightness = np.mean(segment_brightness) if segment_brightness else 100
                brightness_levels.append(avg_brightness)
                clips.append(clip)
            cap.release()
            
            target_brightness = np.median(brightness_levels)
            
            adjusted_clips = []
            for clip, brightness in zip(clips, brightness_levels):
                if brightness > 0:
                    factor = min(max(target_brightness / brightness, 0.5), 2.0)
                    adjusted_clip = clip.fx(vfx.colorx, factor)
                    adjusted_clips.append(adjusted_clip)
                else:
                    adjusted_clips.append(clip)
            
            for i, (task_name, desired_duration) in enumerate(selected_videos, start=1):
                if self.cancel_event.is_set():
                    break
                start_progress = i * task_share
                self.queue.put(("task_start", task_name, start_progress))
                
                total_duration = sum(end - start for start, end in segments)
                if total_duration > 0:
                    speed_factor = total_duration / desired_duration
                    concatenated = concatenate_videoclips(adjusted_clips)
                    sped_up = concatenated.speedx(speed_factor)
                    output_file = f"{base}_{task_name.split()[1]}{ext}"
                    sped_up.write_videofile(output_file, verbose=False, logger=None)
                    output_files[task_name] = output_file
                
                self.queue.put(("task_complete", (i + 1) * task_share))
            
            if self.cancel_event.is_set():
                self.queue.put(("canceled",))
            else:
                self.queue.put(("complete", output_files.get("Generate 60s Video"), output_files.get("Generate 12min Video")))
        except Exception as e:
            self.queue.put(("canceled",))
            print(f"Error during processing: {e}")
        finally:
            video.close()
    
    def compute_motion_scores_with_progress(self, video_path, task_share, start_progress, threshold=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_seconds = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        if total_seconds <= 0:
            cap.release()
            return None
        
        motion_scores = []
        prev_frame = None
        start_time = time.time()
        processed_seconds = 0
        
        while processed_seconds < total_seconds:
            if self.cancel_event.is_set():
                cap.release()
                return None
            
            cap.set(cv2.CAP_PROP_POS_MSEC, processed_seconds * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                motion_score = np.sum(diff > threshold)
                motion_scores.append(motion_score)
            prev_frame = gray
            
            processed_seconds += 1
            progress = start_progress + (processed_seconds / total_seconds) * task_share
            elapsed_time = time.time() - start_time
            if processed_seconds > 0:
                time_per_second = elapsed_time / processed_seconds
                remaining_seconds = max((total_seconds - processed_seconds) * time_per_second, 0)
                self.queue.put(("progress", progress, int(remaining_seconds // 60)))
        
        cap.release()
        return motion_scores
    
    def cancel_processing(self):
        self.cancel_event.set()
    
    def process_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)
    
    def handle_message(self, message):
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
            self.output_60s.configure(text="60s Video: Canceled")
            self.output_12min.configure(text="12min Video: Canceled")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text="Process Canceled")
            self.reset_ui()
    
    def reset_ui(self):
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

if __name__ == "__main__":
    root = ctk.CTk()  # Use CTk instead of Tk for customtkinter root
    app = VideoProcessorApp(root)
    root.mainloop()