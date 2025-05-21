import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
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

class ToggleSwitch(tk.Canvas):
    """Custom toggle switch widget."""
    OFF_TRACK_COLOR = "#D3D3D3"
    ON_TRACK_COLOR = "#008080"
    KNOB_COLOR = "white"
    DISABLED_TRACK_COLOR = "#A9A9A9"
    DISABLED_KNOB_COLOR = "#C0C0C0"
    
    def __init__(self, parent, variable, **kwargs):
        tk.Canvas.__init__(self, parent, width=50, height=25, **kwargs)
        self.variable = variable
        self.bind("<Button-1>", self.toggle)
        self.variable.trace("w", self.on_variable_change)
        self.draw_switch()
    
    def draw_switch(self):
        self.delete("all")
        width = 50
        height = 25
        state = self["state"]
        track_color = self.DISABLED_TRACK_COLOR if state == tk.DISABLED else (self.ON_TRACK_COLOR if self.variable.get() else self.OFF_TRACK_COLOR)
        knob_color = self.DISABLED_KNOB_COLOR if state == tk.DISABLED else self.KNOB_COLOR
        self.create_oval(0, 0, height, height, fill=track_color, outline="")
        self.create_oval(width - height, 0, width, height, fill=track_color, outline="")
        self.create_rectangle(height / 2, 0, width - height / 2, height, fill=track_color, outline="")
        knob_x = height / 2 if not self.variable.get() else width - height / 2
        self.create_oval(knob_x - height / 2, 0, knob_x + height / 2, height, fill=knob_color, outline="")
    
    def toggle(self, event):
        if self["state"] != tk.DISABLED:
            self.variable.set(not self.variable.get())
    
    def on_variable_change(self, *args):
        self.draw_switch()

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Box Video Processor")
        
        self.label = tk.Label(root, text="Select Input Video")
        self.label.pack(pady=10)
        
        self.generate_60s = tk.BooleanVar(value=True)
        tk.Label(root, text="Generate 60s Video").pack()
        self.switch_60s = ToggleSwitch(root, self.generate_60s)
        self.switch_60s.pack(pady=5)
        
        self.generate_12min = tk.BooleanVar(value=True)
        tk.Label(root, text="Generate 12min Video").pack()
        self.switch_12min = ToggleSwitch(root, self.generate_12min)
        self.switch_12min.pack(pady=5)
        
        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        
        self.current_task_label = tk.Label(root, text="Current Task: N/A")
        self.current_task_label.pack(pady=5)
        
        self.time_label = tk.Label(root, text="Estimated Time Remaining: N/A")
        self.time_label.pack(pady=5)
        
        self.cancel_button = tk.Button(root, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(pady=5)
        
        self.output_label = tk.Label(root, text="Output Files:")
        self.output_label.pack(pady=10)
        
        self.output_60s = tk.Label(root, text="60s Video: N/A")
        self.output_60s.pack(pady=5)
        
        self.output_12min = tk.Label(root, text="12min Video: N/A")
        self.output_12min.pack(pady=5)
        
        self.input_file = None
        self.cancel_event = threading.Event()
        self.queue = queue.Queue()
        self.root.after(100, self.process_queue)
    
    def browse_file(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.input_file:
            self.label.config(text=f"Selected: {os.path.basename(self.input_file)}")
            if not self.generate_60s.get() and not self.generate_12min.get():
                messagebox.showwarning("Warning", "Please enable at least one video to generate.")
                return
            self.switch_60s.config(state=tk.DISABLED)
            self.switch_12min.config(state=tk.DISABLED)
            self.browse_button.config(state=tk.DISABLED)
            self.cancel_button.config(state=tk.NORMAL)
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
            for i, (task_name, desired_duration) in enumerate(selected_videos, start=1):
                if self.cancel_event.is_set():
                    break
                start_progress = i * task_share
                self.queue.put(("task_start", task_name, start_progress))
                
                total_duration = sum(end - start for start, end in segments)
                if total_duration > 0:
                    speed_factor = total_duration / desired_duration
                    clips = [video.subclip(start, end) for start, end in segments]
                    concatenated = concatenate_videoclips(clips)
                    sped_up = concatenated.speedx(speed_factor)
                    output_file = f"{base}_{task_name.split()[1]}{ext}"
                    sped_up.write_videofile(output_file, verbose=False, logger=None)
                    output_files[task_name] = output_file
                
                self.queue.put(("task_complete", (i + 1) * task_share))
            
            if self.cancel_event.is_set():
                self.queue.put(("canceled",))
            else:
                self.queue.put(("complete", output_files.get("Generate 60s Video"), output_files.get("Generate 12min Video")))
        finally:
            video.close()
    
    def compute_motion_scores_with_progress(self, video_path, task_share, start_progress, threshold=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_seconds = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0
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
                remaining_seconds = (total_seconds - processed_seconds) * time_per_second
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
            self.progress["value"] = progress_value
            self.time_label.config(text=f"Estimated Time Remaining: {remaining_time} minutes")
        elif message[0] == "task_start":
            task_name, start_progress = message[1:]
            self.current_task_label.config(text=f"Current Task: {task_name}")
            self.progress["value"] = start_progress
            if task_name != "Motion Detection":
                self.time_label.config(text="Processing...")
        elif message[0] == "task_complete":
            progress_value = message[1]
            self.progress["value"] = progress_value
        elif message[0] == "complete":
            output_60s, output_12min = message[1:]
            self.output_60s.config(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_12min.config(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.current_task_label.config(text="Current Task: N/A")
            self.time_label.config(text="Process Complete")
            self.reset_ui()
        elif message[0] == "canceled":
            self.output_60s.config(text="60s Video: Canceled")
            self.output_12min.config(text="12min Video: Canceled")
            self.current_task_label.config(text="Current Task: N/A")
            self.time_label.config(text="Process Canceled")
            self.reset_ui()
    
    def reset_ui(self):
        self.switch_60s.config(state=tk.NORMAL)
        self.switch_12min.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()