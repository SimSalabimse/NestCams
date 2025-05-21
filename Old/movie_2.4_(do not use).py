import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import os
import time
import threading
import queue
import sys
import subprocess

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

class ToolTip:
    """Custom tooltip class for providing helpful hints on widgets."""
    def __init__(self, widget, text, theme):
        self.widget = widget
        self.text = text
        self.theme = theme
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)
    
    def show_tip(self, event):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tip_window, text=self.text, background=self.theme["tooltip_bg"], 
                         foreground=self.theme["tooltip_fg"], relief="solid", borderwidth=1)
        label.pack()
    
    def hide_tip(self, event):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class ToggleSwitch(tk.Canvas):
    """Custom toggle switch widget with theme support and labels."""
    def __init__(self, parent, variable, theme, **kwargs):
        tk.Canvas.__init__(self, parent, width=80, height=40, bg=theme["bg"], highlightthickness=0, **kwargs)
        self.variable = variable
        self.theme = theme
        self.bind("<Button-1>", self.toggle)
        self.variable.trace("w", self.on_variable_change)
        self.draw_switch()
    
    def draw_switch(self):
        self.delete("all")
        width, height = 80, 40
        state = self["state"]
        track_color = self.theme["switch_disabled_track"] if state == tk.DISABLED else (self.theme["switch_on"] if self.variable.get() else self.theme["switch_off"])
        knob_color = self.theme["switch_disabled_knob"] if state == tk.DISABLED else self.theme["switch_knob"]
        
        # Draw track
        self.create_oval(0, 0, height, height, fill=track_color, outline="")
        self.create_oval(width - height, 0, width, height, fill=track_color, outline="")
        self.create_rectangle(height / 2, 0, width - height / 2, height, fill=track_color, outline="")
        
        # Draw knob
        knob_x = height / 2 if not self.variable.get() else width - height / 2
        self.create_oval(knob_x - height / 2, 0, knob_x + height / 2, height, fill=knob_color, outline="")
        
        # Draw "On" and "Off" text
        text_color = self.theme["switch_text"]
        if self.variable.get():
            self.create_text(width - 15, height / 2, text="On", fill=text_color, font=("Arial", 12))
        else:
            self.create_text(15, height / 2, text="Off", fill=text_color, font=("Arial", 12))
    
    def toggle(self, event):
        if self["state"] != tk.DISABLED:
            self.variable.set(not self.variable.get())
    
    def on_variable_change(self, *args):
        self.draw_switch()
    
    def set_theme(self, theme):
        self.theme = theme
        self.config(bg=theme["bg"])
        self.draw_switch()

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Box Video Processor")
        self.root.geometry("500x700")
        
        # Define themes with improved colors
        self.dark_theme = {
            "bg": "#2C2C2C",
            "fg": "#FFFFFF",
            "button_bg": "#3C3C3C",
            "button_fg": "#FFFFFF",
            "progress_trough": "#2C2C2C",
            "progress_bar": "#008080",
            "switch_off": "#555555",
            "switch_on": "#00A0A0",
            "switch_knob": "#FFFFFF",
            "switch_text": "#FFFFFF",
            "switch_disabled_track": "#3C3C3C",
            "switch_disabled_knob": "#5C5C5C",
            "separator": "#3C3C3C",
            "tooltip_bg": "#333333",
            "tooltip_fg": "#FFFFFF",
        }
        self.light_theme = {
            "bg": "#FFFFFF",
            "fg": "#000000",
            "button_bg": "#E0E0E0",
            "button_fg": "#000000",
            "progress_trough": "#FFFFFF",
            "progress_bar": "#008080",
            "switch_off": "#AAAAAA",
            "switch_on": "#008080",
            "switch_knob": "#FFFFFF",
            "switch_text": "#000000",
            "switch_disabled_track": "#A9A9A9",
            "switch_disabled_knob": "#C0C0C0",
            "separator": "#E0E0E0",
            "tooltip_bg": "#FFFFDD",
            "tooltip_fg": "#000000",
        }
        self.dark_mode = True
        self.current_theme = self.dark_theme if self.dark_mode else self.light_theme
        
        # Configure styles
        self.style = ttk.Style()
        self.update_theme_styles()
        
        # Set root background
        self.root.configure(bg=self.current_theme["bg"])
        
        # Title
        self.title_label = ttk.Label(self.root, text="Bird Box Video Processor", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)
        
        # Input section
        self.input_frame = tk.Frame(self.root, bg=self.current_theme["bg"])
        self.input_frame.pack(pady=10)
        self.label = ttk.Label(self.input_frame, text="Select Input Video")
        self.label.pack(side="left", padx=5)
        self.browse_button = ttk.Button(self.input_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side="left", padx=5)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)
        
        # Output directory selection
        self.output_dir_frame = tk.Frame(self.root, bg=self.current_theme["bg"])
        self.output_dir_frame.pack(pady=10)
        ttk.Label(self.output_dir_frame, text="Output Directory:").pack(side="left")
        self.output_dir_var = tk.StringVar()
        self.output_dir_entry = tk.Entry(self.output_dir_frame, textvariable=self.output_dir_var, width=40)
        self.output_dir_entry.pack(side="left", padx=5)
        self.output_dir_button = ttk.Button(self.output_dir_frame, text="Browse", command=self.select_output_dir)
        self.output_dir_button.pack(side="left")
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)
        
        # Output options
        self.output_options_frame = tk.Frame(self.root, bg=self.current_theme["bg"])
        self.output_options_frame.pack(pady=10)
        self.generate_60s = tk.BooleanVar(value=True)
        ttk.Label(self.output_options_frame, text="Generate 60s Video").pack()
        self.switch_60s = ToggleSwitch(self.output_options_frame, self.generate_60s, theme=self.current_theme)
        self.switch_60s.pack(pady=5)
        self.generate_5min = tk.BooleanVar(value=False)
        ttk.Label(self.output_options_frame, text="Generate 5min Video").pack()
        self.switch_5min = ToggleSwitch(self.output_options_frame, self.generate_5min, theme=self.current_theme)
        self.switch_5min.pack(pady=5)
        self.generate_12min = tk.BooleanVar(value=True)
        ttk.Label(self.output_options_frame, text="Generate 12min Video").pack()
        self.switch_12min = ToggleSwitch(self.output_options_frame, self.generate_12min, theme=self.current_theme)
        self.switch_12min.pack(pady=5)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)
        
        # Processing controls
        self.processing_frame = tk.Frame(self.root, bg=self.current_theme["bg"])
        self.processing_frame.pack(pady=10)
        self.start_button = ttk.Button(self.processing_frame, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.start_button.pack(side="left", padx=5)
        self.cancel_button = ttk.Button(self.processing_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side="left", padx=5)
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)
        self.current_task_label = ttk.Label(self.root, text="Current Task: N/A")
        self.current_task_label.pack(pady=5)
        self.time_label = ttk.Label(self.root, text="Estimated Time Remaining: N/A")
        self.time_label.pack(pady=5)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)
        
        # Output display
        self.output_display_frame = tk.Frame(self.root, bg=self.current_theme["bg"])
        self.output_display_frame.pack(pady=10)
        self.output_label = ttk.Label(self.output_display_frame, text="Output Files:")
        self.output_label.pack(pady=5)
        self.output_60s = ttk.Label(self.output_display_frame, text="60s Video: N/A")
        self.output_60s.pack(pady=2)
        self.output_5min = ttk.Label(self.output_display_frame, text="5min Video: N/A")
        self.output_5min.pack(pady=2)
        self.output_12min = ttk.Label(self.output_display_frame, text="12min Video: N/A")
        self.output_12min.pack(pady=2)
        self.open_dir_button = ttk.Button(self.output_display_frame, text="Open Output Directory", command=self.open_output_directory, state=tk.DISABLED)
        self.open_dir_button.pack(pady=5)
        
        # Settings button
        self.settings_button = ttk.Button(self.root, text="Settings", command=self.open_settings)
        self.settings_button.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        self.input_file = None
        self.cancel_event = threading.Event()
        self.queue = queue.Queue()
        self.root.after(100, self.process_queue)
        
        # Default settings
        self.motion_threshold = 30
        
        # Tooltips
        ToolTip(self.switch_60s, "Enable to generate a 60-second highlight clip", self.current_theme)
        ToolTip(self.switch_5min, "Enable to generate a 5-minute highlight clip", self.current_theme)
        ToolTip(self.switch_12min, "Enable to generate a 12-minute highlight clip", self.current_theme)
        ToolTip(self.browse_button, "Select the input video file", self.current_theme)
        ToolTip(self.output_dir_button, "Select the directory to save output files", self.current_theme)
        ToolTip(self.start_button, "Start processing the video", self.current_theme)
        ToolTip(self.cancel_button, "Cancel the ongoing processing", self.current_theme)
        ToolTip(self.settings_button, "Open settings to adjust motion threshold and theme", self.current_theme)
        
        # Trace variables for dynamic state updates
        self.generate_60s.trace("w", lambda *args: self.update_start_button_state())
        self.generate_5min.trace("w", lambda *args: self.update_start_button_state())
        self.generate_12min.trace("w", lambda *args: self.update_start_button_state())
    
    def update_theme_styles(self):
        self.style.configure("TLabel", background=self.current_theme["bg"], foreground=self.current_theme["fg"], font=("Arial", 12))
        self.style.configure("TButton", background=self.current_theme["button_bg"], foreground=self.current_theme["button_fg"], font=("Arial", 12))
        self.style.configure("TProgressbar", troughcolor=self.current_theme["progress_trough"], background=self.current_theme["progress_bar"])
        self.style.configure("TSeparator", background=self.current_theme["separator"])
    
    def update_theme(self):
        self.current_theme = self.dark_theme if self.dark_mode else self.light_theme
        self.root.configure(bg=self.current_theme["bg"])
        self.update_theme_styles()
        self.switch_60s.set_theme(self.current_theme)
        self.switch_5min.set_theme(self.current_theme)
        self.switch_12min.set_theme(self.current_theme)
        self.input_frame.configure(bg=self.current_theme["bg"])
        self.output_dir_frame.configure(bg=self.current_theme["bg"])
        self.output_options_frame.configure(bg=self.current_theme["bg"])
        self.processing_frame.configure(bg=self.current_theme["bg"])
        self.output_display_frame.configure(bg=self.current_theme["bg"])
    
    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("300x200")
        settings_window.configure(bg=self.current_theme["bg"])
        
        ttk.Label(settings_window, text="Motion Detection Threshold").pack(pady=5)
        threshold_scale = ttk.Scale(settings_window, from_=10, to=100, orient="horizontal", length=200, value=self.motion_threshold)
        threshold_scale.pack(pady=5)
        
        ttk.Label(settings_window, text="Dark Mode").pack(pady=5)
        dark_mode_var = tk.BooleanVar(value=self.dark_mode)
        dark_mode_switch = ToggleSwitch(settings_window, dark_mode_var, theme=self.current_theme)
        dark_mode_switch.pack(pady=5)
        
        def save_settings():
            self.motion_threshold = int(threshold_scale.get())
            self.dark_mode = dark_mode_var.get()
            self.update_theme()
            settings_window.destroy()
        
        ttk.Button(settings_window, text="Save", command=save_settings).pack(pady=10)
    
    def browse_file(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.input_file:
            self.label.config(text=f"Selected: {os.path.basename(self.input_file)}")
            self.update_start_button_state()
    
    def select_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def update_start_button_state(self):
        if self.input_file and (self.generate_60s.get() or self.generate_5min.get() or self.generate_12min.get()):
            self.start_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.DISABLED)
    
    def start_processing(self):
        self.switch_60s.config(state=tk.DISABLED)
        self.switch_5min.config(state=tk.DISABLED)
        self.switch_12min.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.cancel_event.clear()
        self.worker_thread = threading.Thread(target=self.process_video_thread)
        self.worker_thread.start()
    
    def process_video_thread(self):
        selected_videos = []
        if self.generate_60s.get():
            selected_videos.append(("Generate 60s Video", 60))
        if self.generate_5min.get():
            selected_videos.append(("Generate 5min Video", 300))
        if self.generate_12min.get():
            selected_videos.append(("Generate 12min Video", 720))
        
        total_tasks = 1 + len(selected_videos)
        task_share = 100 / total_tasks
        
        self.queue.put(("task_start", "Motion Detection", 0))
        motion_scores = self.compute_motion_scores_with_progress(self.input_file, task_share, 0, self.motion_threshold)
        if motion_scores is None or self.cancel_event.is_set():
            self.queue.put(("canceled",))
            return
        
        segments = find_motion_segments(motion_scores)
        if not segments:
            self.queue.put(("complete", "No motion detected", "No motion detected", "No motion detected"))
            return
        
        base = os.path.splitext(os.path.basename(self.input_file))[0]
        ext = os.path.splitext(self.input_file)[1]
        output_dir = self.output_dir_var.get() if self.output_dir_var.get() else os.path.dirname(self.input_file)
        output_files = {}
        video = VideoFileClip(self.input_file)
        
        try:
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
                    output_file = os.path.join(output_dir, f"{base}_{task_name.split()[1]}{ext}")
                    sped_up.write_videofile(output_file, verbose=False, logger=None)
                    output_files[task_name] = output_file
                
                self.queue.put(("task_complete", (i + 1) * task_share))
            
            if self.cancel_event.is_set():
                self.queue.put(("canceled",))
            else:
                self.queue.put(("complete", output_files.get("Generate 60s Video"), output_files.get("Generate 5min Video"), output_files.get("Generate 12min Video")))
        except Exception as e:
            self.queue.put(("canceled",))
            print(f"Error during processing: {e}")
        finally:
            video.close()
    
    def compute_motion_scores_with_progress(self, video_path, task_share, start_progress, threshold):
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
            self.progress["value"] = progress_value
            self.time_label.config(text=f"Estimated Time Remaining: {remaining_time} minutes")
        elif message[0] == "task_start":
            task_name, start_progress = message[1:]
            self.current_task_label.config(text=f"Current Task: {task_name}")
            self.progress["value"] = start_progress
            if task_name != "Motion Detection":
                self.time_label.config(text="Processing...")
            self.status_label.config(text=f"Processing: {task_name}")
        elif message[0] == "task_complete":
            progress_value = message[1]
            self.progress["value"] = progress_value
        elif message[0] == "complete":
            output_60s, output_5min, output_12min = message[1:]
            self.output_60s.config(text=f"60s Video: {output_60s if output_60s else 'Not generated'}")
            self.output_5min.config(text=f"5min Video: {output_5min if output_5min else 'Not generated'}")
            self.output_12min.config(text=f"12min Video: {output_12min if output_12min else 'Not generated'}")
            self.current_task_label.config(text="Current Task: N/A")
            self.time_label.config(text="Process Complete")
            self.status_label.config(text="Processing complete")
            if output_60s or output_5min or output_12min:
                self.open_dir_button.config(state=tk.NORMAL)
            self.reset_ui()
        elif message[0] == "canceled":
            self.output_60s.config(text="60s Video: Canceled")
            self.output_5min.config(text="5min Video: Canceled")
            self.output_12min.config(text="12min Video: Canceled")
            self.current_task_label.config(text="Current Task: N/A")
            self.time_label.config(text="Process Canceled")
            self.status_label.config(text="Processing canceled")
            self.reset_ui()
    
    def reset_ui(self):
        self.switch_60s.config(state=tk.NORMAL)
        self.switch_5min.config(state=tk.NORMAL)
        self.switch_12min.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.update_start_button_state()
    
    def open_output_directory(self):
        output_dir = self.output_dir_var.get() if self.output_dir_var.get() else os.path.dirname(self.input_file)
        if sys.platform == "win32":
            os.startfile(output_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", output_dir])
        else:
            subprocess.run(["xdg-open", output_dir])

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()