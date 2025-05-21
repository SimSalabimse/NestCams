import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import time
import threading
import queue

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

class ToggleSwitch(tk.Canvas):
    """Custom toggle switch widget mimicking the described design."""
    OFF_TRACK_COLOR = "#D3D3D3"  # Light gray when off
    ON_TRACK_COLOR = "#008080"   # Teal when on
    KNOB_COLOR = "white"         # White knob
    DISABLED_TRACK_COLOR = "#A9A9A9"  # Gray when disabled
    DISABLED_KNOB_COLOR = "#C0C0C0"   # Light gray knob when disabled
    
    def __init__(self, parent, variable, **kwargs):
        tk.Canvas.__init__(self, parent, width=50, height=25, **kwargs)
        self.variable = variable
        self.bind("<Button-1>", self.toggle)
        self.variable.trace("w", self.on_variable_change)
        self.draw_switch()
    
    def draw_switch(self):
        """Draw the switch based on its state."""
        self.delete("all")
        width = 50  # Fixed width
        height = 25 # Fixed height
        
        state = self["state"]
        if state == tk.DISABLED:
            track_color = self.DISABLED_TRACK_COLOR
            knob_color = self.DISABLED_KNOB_COLOR
        else:
            track_color = self.OFF_TRACK_COLOR if not self.variable.get() else self.ON_TRACK_COLOR
            knob_color = self.KNOB_COLOR
        
        # Draw track (rounded rectangle)
        self.create_oval(0, 0, height, height, fill=track_color, outline="")
        self.create_oval(width - height, 0, width, height, fill=track_color, outline="")
        self.create_rectangle(height / 2, 0, width - height / 2, height, fill=track_color, outline="")
        
        # Draw knob
        knob_x = height / 2 if not self.variable.get() else width - height / 2
        self.create_oval(knob_x - height / 2, 0, knob_x + height / 2, height, fill=knob_color, outline="")
    
    def toggle(self, event):
        """Toggle the switch state on click if not disabled."""
        if self["state"] != tk.DISABLED:
            self.variable.set(not self.variable.get())
    
    def on_variable_change(self, *args):
        """Redraw the switch when the variable changes."""
        self.draw_switch()

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize the GUI window and components."""
        self.root = root
        self.root.title("Bird Box Video Processor")
        
        # File selection
        self.label = tk.Label(root, text="Select Input Video")
        self.label.pack(pady=10)
        
        # Toggle switches for video output selection
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
        """Open a file dialog to select a video and start processing."""
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
        """Run the video processing in a separate thread."""
        if self.cancel_event.is_set():
            return
        motion_scores = self.compute_motion_scores_with_progress(self.input_file)
        if motion_scores is None or self.cancel_event.is_set():
            self.queue.put(("canceled",))
            return
        base, ext = os.path.splitext(self.input_file)
        output_60s = None
        output_12min = None
        if self.generate_60s.get() and not self.cancel_event.is_set():
            start_times_60s = select_top_segments(motion_scores, 12, 5)
            output_60s = f"{base}_60s{ext}"
            self.generate_short_video(self.input_file, output_60s, start_times_60s, 12)
        if self.generate_12min.get() and not self.cancel_event.is_set():
            start_times_12min = select_top_segments(motion_scores, 72, 10)
            output_12min = f"{base}_12min{ext}"
            self.generate_short_video(self.input_file, output_12min, start_times_12min, 72)
        if self.cancel_event.is_set():
            self.queue.put(("canceled",))
        else:
            self.queue.put(("complete", output_60s, output_12min))
    
    def compute_motion_scores_with_progress(self, video_path, threshold=30):
        """Compute motion scores for each second of the video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        processed_frames = 0
        start_time = time.time()
        motion_scores = []
        prev_frame = None
        for t in range(total_frames):
            if self.cancel_event.is_set():
                cap.release()
                return None
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
            processed_frames += 1
            progress = (processed_frames / total_frames) * 100
            elapsed_time = time.time() - start_time
            if processed_frames > 0:
                time_per_frame = elapsed_time / processed_frames
                remaining_frames = total_frames - processed_frames
                remaining_time = remaining_frames * time_per_frame
                self.queue.put(("progress", progress, int(remaining_time // 60)))
        cap.release()
        return motion_scores
    
    def generate_short_video(self, input_video, output_file, start_times, clip_length):
        """Generate a short video from selected segments."""
        video = VideoFileClip(input_video)
        clips = [video.subclip(t, t + clip_length) for t in start_times]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_file, verbose=False)
        video.close()
    
    def cancel_processing(self):
        """Set the cancel event to stop processing."""
        self.cancel_event.set()
    
    def process_queue(self):
        """Process messages from the queue to update the GUI."""
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)
    
    def handle_message(self, message):
        """Handle messages from the worker thread."""
        if message[0] == "progress":
            progress_value, remaining_time = message[1:]
            self.progress["value"] = progress_value
            self.time_label.config(text=f"Estimated Time Remaining: {remaining_time} minutes")
        elif message[0] == "complete":
            output_60s, output_12min = message[1:]
            self.output_60s.config(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_12min.config(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.switch_60s.config(state=tk.NORMAL)
            self.switch_12min.config(state=tk.NORMAL)
            self.browse_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
        elif message[0] == "canceled":
            self.output_60s.config(text="60s Video: Canceled")
            self.output_12min.config(text="12min Video: Canceled")
            self.switch_60s.config(state=tk.NORMAL)
            self.switch_12min.config(state=tk.NORMAL)
            self.browse_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()