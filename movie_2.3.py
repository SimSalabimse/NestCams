import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, VideoClip
import os
import time
import threading
import queue

# [Previous code for find_motion_segments and ToggleSwitch remains unchanged...]

class VideoProcessorApp:
    def __init__(self, root):
        # [Existing initialization code...]
        pass

    def browse_file(self):
        # [Existing browse_file code...]
        pass

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
                    # Apply brightness adjustment to each segment
                    clips = [self.adjust_brightness(video.subclip(start, end)) for start, end in segments]
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

    def adjust_brightness(self, clip):
        """Adjust the brightness of each frame in the clip using histogram equalization."""
        def process_frame(get_frame, t):
            frame = get_frame(t)
            if frame.ndim == 3 and frame.shape[2] == 3:  # Ensure it's a color frame
                # Convert to grayscale for histogram equalization
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # Apply histogram equalization to normalize brightness
                equalized = cv2.equalizeHist(gray)
                # Apply the equalized brightness to all channels
                frame[:, :, 0] = equalized
                frame[:, :, 1] = equalized
                frame[:, :, 2] = equalized
            return frame
        
        # Return a new VideoClip with adjusted frames
        return VideoClip(make_frame=lambda t: process_frame(clip.get_frame, t), duration=clip.duration)

    def compute_motion_scores_with_progress(self, video_path, task_share, start_progress, threshold=30):
        # [Existing motion detection code...]
        pass

    # [Remaining methods like cancel_processing, process_queue, handle_message, reset_ui remain unchanged...]

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()