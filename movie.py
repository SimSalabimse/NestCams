import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import time

def select_top_segments(motion_scores, segment_length, num_segments):
    """Select the top segments with the highest motion scores."""
    num_chunks = len(motion_scores) // segment_length
    chunk_sums = []
    
    for i in range(num_chunks):
        chunk_sum = sum(motion_scores[i * segment_length : (i + 1) * segment_length])
        chunk_sums.append((i, chunk_sum))
    
    chunk_sums.sort(key=lambda x: x[1], reverse=True)
    top_chunks = chunk_sums[:num_segments]
    top_chunks.sort(key=lambda x: x[0])
    start_times = [i * segment_length for i, _ in top_chunks]
    return start_times

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize the GUI window and components."""
        self.root = root
        self.root.title("Bird Box Video Processor")
        
        # GUI components
        self.label = tk.Label(root, text="Select Input Video")
        self.label.pack(pady=10)
        
        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        
        self.time_label = tk.Label(root, text="Estimated Time Remaining: N/A")
        self.time_label.pack(pady=5)
        
        self.output_label = tk.Label(root, text="Output Files:")
        self.output_label.pack(pady=10)
        
        self.output_60s = tk.Label(root, text="60s Video: N/A")
        self.output_60s.pack(pady=5)
        
        self.output_12min = tk.Label(root, text="12min Video: N/A")
        self.output_12min.pack(pady=5)
        
        self.input_file = None
        self.start_time = None
        self.total_frames = 0
        self.processed_frames = 0
    
    def browse_file(self):
        """Open a file dialog to select a video and start processing."""
        self.input_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.input_file:
            self.label.config(text=f"Selected: {os.path.basename(self.input_file)}")
            self.process_video()
    
    def process_video(self):
        """Process the video: compute motion scores and generate output videos."""
        if not self.input_file:
            return
        
        # Reset GUI elements
        self.progress["value"] = 0
        self.time_label.config(text="Estimated Time Remaining: Calculating...")
        self.output_60s.config(text="60s Video: Processing...")
        self.output_12min.config(text="12min Video: Processing...")
        
        # Compute motion scores with progress updates
        motion_scores = self.compute_motion_scores_with_progress(self.input_file)
        
        # Select top segments for 60s (5 segments of 12s) and 12min (10 segments of 72s)
        start_times_60s = select_top_segments(motion_scores, 12, 5)
        start_times_12min = select_top_segments(motion_scores, 72, 10)
        
        # Define output file paths
        base, ext = os.path.splitext(self.input_file)
        output_60s = f"{base}_60s{ext}"
        output_12min = f"{base}_12min{ext}"
        
        # Generate the short videos
        self.generate_short_video_with_progress(self.input_file, output_60s, start_times_60s, 12)
        self.generate_short_video_with_progress(self.input_file, output_12min, start_times_12min, 72)
        
        # Update output labels with file paths
        self.output_60s.config(text=f"60s Video: {output_60s}")
        self.output_12min.config(text=f"12min Video: {output_12min}")
    
    def compute_motion_scores_with_progress(self, video_path, threshold=30):
        """Compute motion scores for each second of the video and update progress."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        self.processed_frames = 0
        self.start_time = time.time()
        
        motion_scores = []
        prev_frame = None
        
        for t in range(self.total_frames):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                motion_score = np.sum(diff > threshold)
                motion_scores.append(motion_score)
            prev_frame = gray
            
            # Update progress bar and time remaining
            self.processed_frames += 1
            progress = (self.processed_frames / self.total_frames) * 100
            self.progress["value"] = progress
            
            elapsed_time = time.time() - self.start_time
            if self.processed_frames > 0:
                time_per_frame = elapsed_time / self.processed_frames
                remaining_frames = self.total_frames - self.processed_frames
                remaining_time = remaining_frames * time_per_frame
                self.time_label.config(text=f"Estimated Time Remaining: {int(remaining_time // 60)} minutes")
            
            self.root.update_idletasks()  # Update GUI in real-time
        
        cap.release()
        return motion_scores
    
    def generate_short_video_with_progress(self, input_video, output_file, start_times, clip_length):
        """Generate a short video from selected segments."""
        video = VideoFileClip(input_video)
        clips = [video.subclip(t, t + clip_length) for t in start_times]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_file, verbose=False)
        video.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()