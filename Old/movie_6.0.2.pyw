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
import json
from PIL import Image

# Version number
VERSION = "6.0.2"  # Updated with Mini Video Player

### Helper Functions

def compute_motion_score(prev_frame, current_frame, threshold=30):
    """Compute motion score between two frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.sum(diff > threshold, dtype=np.uint32)

def is_white_or_black_frame(frame, white_threshold=200, black_threshold=50):
    """Check if frame is predominantly white or black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > white_threshold or mean_brightness < black_threshold

def normalize_frame(frame, clip_limit=1.0, saturation_multiplier=1.1):
    """Normalize frame brightness and enhance saturation."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = (s.astype('float32') * saturation_multiplier).clip(0, 255).astype('uint8')
        hsv_enhanced = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    except MemoryError:
        return None

def process_video_multi_pass(input_path, output_path, desired_duration, motion_threshold=3000, sample_interval=5, white_threshold=200, black_threshold=50, clip_limit=1.0, saturation_multiplier=1.1, output_format='mp4', progress_callback=None, cancel_event=None):
    """Process video in multiple passes."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return "Failed to open video file"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if total_frames <= 0 or fps <= 0:
        return "Invalid video properties"
    
    rotate = desired_duration == 60
    format_codecs = {'mp4': 'mp4v', 'avi': 'XVID', 'mkv': 'X264'}
    fourcc = cv2.VideoWriter_fourcc(*format_codecs.get(output_format, 'mp4v'))
    
    temp_path1 = f"temp_intermediate_{uuid.uuid4().hex}.{output_format}"
    in_motion = False
    include_indices = []
    prev_frame = None
    start_time = time.time()
    
    cap = cv2.VideoCapture(input_path)
    for frame_idx in range(total_frames):
        if cancel_event and cancel_event.is_set():
            cap.release()
            return "Processing canceled by user"
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0 and prev_frame is not None:
            motion_score = compute_motion_score(prev_frame, frame, threshold=motion_threshold)
            if motion_score > motion_threshold:
                in_motion = True
                include_indices.append(frame_idx)
            elif motion_score <= motion_threshold and in_motion:
                in_motion = False
            prev_frame = frame
        elif in_motion:
            include_indices.append(frame_idx)
        
        if progress_callback and frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
            progress_callback(frame_idx / total_frames * 33, frame_idx, total_frames, remaining)
    
    cap.release()
    
    if not include_indices:
        target_frames = int(desired_duration * fps)
        include_indices = [i for i in range(0, total_frames, max(1, total_frames // target_frames))]
    
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(temp_path1, fourcc, fps, (frame_width, frame_height))
    for idx, frame_idx in enumerate(include_indices):
        if cancel_event and cancel_event.is_set():
            cap.release()
            out.release()
            os.remove(temp_path1)
            return "Processing canceled by user"
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        if progress_callback and idx % 50 == 0:
            progress_callback(33 + (idx / len(include_indices) * 33), idx, len(include_indices), 0)
    cap.release()
    out.release()
    
    temp_path2 = f"temp_filtered_{uuid.uuid4().hex}.{output_format}"
    cap = cv2.VideoCapture(temp_path1)
    out = cv2.VideoWriter(temp_path2, fourcc, fps, (frame_width, frame_height))
    filtered_indices = []
    
    total_intermediate_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(total_intermediate_frames):
        if cancel_event and cancel_event.is_set():
            cap.release()
            out.release()
            os.remove(temp_path1)
            return "Processing canceled by user"
        ret, frame = cap.read()
        if not ret:
            break
        if not is_white_or_black_frame(frame, white_threshold, black_threshold):
            out.write(frame)
            filtered_indices.append(frame_idx)
        
        if progress_callback and frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_intermediate_frames - frame_idx) / rate if rate > 0 else 0
            progress_callback(33 + (frame_idx / total_intermediate_frames * 33), frame_idx, total_intermediate_frames, remaining)
    
    cap.release()
    out.release()
    os.remove(temp_path1)
    
    target_frames = int(desired_duration * fps)
    cap = cv2.VideoCapture(temp_path2)
    
    if rotate:
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_height, frame_width))
    else:
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
            normalized_frame = normalize_frame(frame, clip_limit, saturation_multiplier)
            if normalized_frame is None:
                cap.release()
                out.release()
                os.remove(temp_path2)
                return "Memory error during processing"
            if rotate:
                normalized_frame = cv2.rotate(normalized_frame, cv2.ROTATE_90_CLOCKWISE)
            out.write(normalized_frame)
        
        if progress_callback and idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(final_indices) - idx) / rate if rate > 0 else 0
            progress_callback(66 + (idx / len(final_indices) * 34), idx, len(final_indices), remaining)
    
    cap.release()
    out.release()
    os.remove(temp_path2)
    return None if final_indices else "No frames written after trimming"

### Main Application Class

class VideoProcessorApp:
    def __init__(self, root):
        """Initialize GUI with Mini Video Player."""
        self.root = root
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.theme_var = tk.StringVar(value="Dark")
        theme_frame = ctk.CTkFrame(root)
        theme_frame.pack(pady=5)
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(theme_frame, variable=self.theme_var, values=["Light", "Dark"], command=self.toggle_theme).pack(side=tk.LEFT)
        
        self.label = ctk.CTkLabel(root, text="Select Input Video(s)")
        self.label.pack(pady=10)
        
        self.generate_60s = tk.BooleanVar(value=True)
        self.switch_60s = ctk.CTkSwitch(root, text="Generate 60s Video", variable=self.generate_60s)
        self.switch_60s.pack(pady=5)
        
        self.generate_12min = tk.BooleanVar(value=True)
        self.switch_12min = ctk.CTkSwitch(root, text="Generate 12min Video", variable=self.generate_12min)
        self.switch_12min.pack(pady=5)
        
        self.output_format_var = tk.StringVar(value="mp4")
        format_frame = ctk.CTkFrame(root)
        format_frame.pack(pady=5)
        ctk.CTkLabel(format_frame, text="Output Format:").pack(side=tk.LEFT, padx=5)
        ctk.CTkOptionMenu(format_frame, variable=self.output_format_var, values=["mp4", "avi", "mkv"]).pack(side=tk.LEFT)
        
        self.settings_button = ctk.CTkButton(root, text="Settings & Preview", command=self.open_settings)
        self.settings_button.pack(pady=5)
        
        self.browse_button = ctk.CTkButton(root, text="Browse", command=self.browse_files)
        self.browse_button.pack(pady=5)
        
        self.start_button = ctk.CTkButton(root, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=5)
        
        self.progress = ctk.CTkProgressBar(root, width=300)
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
        
        self.output_text = ctk.CTkTextbox(root, height=100, width=400)
        self.output_text.pack(pady=5)
        self.output_text.insert("0.0", "No outputs yet.")
        
        self.input_files = []
        self.cancel_event = threading.Event()
        self.queue = queue.Queue()
        self.start_time = None
        self.preview_image = None  # Reference for CTkImage
        self.root.after(100, self.process_queue)
        
        # Default settings
        self.motion_threshold = 3000
        self.white_threshold = 200
        self.black_threshold = 50
        self.clip_limit = 1.0
        self.saturation_multiplier = 1.1
        
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.motion_threshold = int(settings.get("motion_threshold", 3000))
                self.white_threshold = int(settings.get("white_threshold", 200))
                self.black_threshold = int(settings.get("black_threshold", 50))
                self.clip_limit = float(settings.get("clip_limit", 1.0))
                self.saturation_multiplier = float(settings.get("saturation_multiplier", 1.1))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            print("Warning: Could not load settings, using defaults")
        
        self.presets = {}
        try:
            with open("presets.json", "r") as f:
                self.presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("No presets found")
    
    def toggle_theme(self, theme):
        """Switch between light and dark themes."""
        ctk.set_appearance_mode(theme)
    
    def open_settings(self):
        """Open settings window with Mini Video Player."""
        self.settings_window = ctk.CTkToplevel(self.root)
        self.settings_window.title("Settings & Preview")
        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_settings_close)
        
        settings_frame = ctk.CTkFrame(self.settings_window)
        settings_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        motion_label = ctk.CTkLabel(settings_frame, text="Motion Sensitivity")
        motion_label.pack(pady=5)
        self.motion_slider = ctk.CTkSlider(settings_frame, from_=500, to=20000, number_of_steps=195, command=self.update_settings)
        self.motion_slider.set(self.motion_threshold)
        self.motion_slider.pack(pady=5)
        self.motion_value_label = ctk.CTkLabel(settings_frame, text=f"Threshold: {self.motion_threshold}")
        self.motion_value_label.pack(pady=5)
        
        white_label = ctk.CTkLabel(settings_frame, text="White Threshold")
        white_label.pack(pady=2)
        self.white_slider = ctk.CTkSlider(settings_frame, from_=100, to=255, number_of_steps=155, command=self.update_settings)
        self.white_slider.set(self.white_threshold)
        self.white_slider.pack(pady=2)
        self.white_value_label = ctk.CTkLabel(settings_frame, text=f"White: {self.white_threshold}")
        self.white_value_label.pack(pady=2)
        
        black_label = ctk.CTkLabel(settings_frame, text="Black Threshold")
        black_label.pack(pady=2)
        self.black_slider = ctk.CTkSlider(settings_frame, from_=0, to=100, number_of_steps=100, command=self.update_settings)
        self.black_slider.set(self.black_threshold)
        self.black_slider.pack(pady=2)
        self.black_value_label = ctk.CTkLabel(settings_frame, text=f"Black: {self.black_threshold}")
        self.black_value_label.pack(pady=2)
        
        clip_label = ctk.CTkLabel(settings_frame, text="CLAHE Clip Limit")
        clip_label.pack(pady=2)
        self.clip_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=5.0, number_of_steps=90, command=self.update_settings)
        self.clip_slider.set(self.clip_limit)
        self.clip_slider.pack(pady=2)
        self.clip_value_label = ctk.CTkLabel(settings_frame, text=f"Clip Limit: {self.clip_limit:.1f}")
        self.clip_value_label.pack(pady=2)
        
        saturation_label = ctk.CTkLabel(settings_frame, text="Saturation Multiplier")
        saturation_label.pack(pady=2)
        self.saturation_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=2.0, number_of_steps=150, command=self.update_settings)
        self.saturation_slider.set(self.saturation_multiplier)
        self.saturation_slider.pack(pady=2)
        self.saturation_value_label = ctk.CTkLabel(settings_frame, text=f"Saturation: {self.saturation_multiplier:.1f}")
        self.saturation_value_label.pack(pady=2)
        
        preset_frame = ctk.CTkFrame(settings_frame)
        preset_frame.pack(pady=10)
        ctk.CTkLabel(preset_frame, text="Preset Management").pack(pady=5)
        
        self.preset_combobox = ctk.CTkComboBox(preset_frame, values=list(self.presets.keys()))
        self.preset_combobox.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Load Preset", command=self.load_preset).pack(pady=5)
        
        self.preset_name_entry = ctk.CTkEntry(preset_frame, placeholder_text="Enter preset name")
        self.preset_name_entry.pack(pady=2)
        ctk.CTkButton(preset_frame, text="Save Preset", command=self.save_preset).pack(pady=5)
        
        ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        ctk.CTkButton(settings_frame, text="Reset to Default", command=self.reset_to_default).pack(pady=10)
        
        # Preview Frame with Mini Video Player
        self.preview_frame = ctk.CTkFrame(self.settings_window)
        self.preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Select a video to enable preview")
        self.preview_label.pack(pady=5)
        
        # Playback controls
        control_frame = ctk.CTkFrame(self.preview_frame)
        control_frame.pack(pady=5)
        self.play_button = ctk.CTkButton(control_frame, text="Play Preview", command=self.start_preview_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ctk.CTkButton(control_frame, text="Stop Preview", command=self.stop_preview_playback, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.preview_cap = None
        self.playback_thread = None
        self.preview_running = threading.Event()
        if self.input_files:
            self.preview_cap = cv2.VideoCapture(self.input_files[0])
            self.start_preview_playback()  # Auto-start preview for convenience
    
    def update_settings(self, value):
        """Update settings from sliders."""
        self.motion_threshold = int(self.motion_slider.get())
        self.white_threshold = int(self.white_slider.get())
        self.black_threshold = int(self.black_slider.get())
        self.clip_limit = float(self.clip_slider.get())
        self.saturation_multiplier = float(self.saturation_slider.get())
        
        self.motion_value_label.configure(text=f"Threshold: {self.motion_threshold}")
        self.white_value_label.configure(text=f"White: {self.white_threshold}")
        self.black_value_label.configure(text=f"Black: {self.black_threshold}")
        self.clip_value_label.configure(text=f"Clip Limit: {self.clip_limit:.1f}")
        self.saturation_value_label.configure(text=f"Saturation: {self.saturation_multiplier:.1f}")
    
    def start_preview_playback(self):
        """Start the Mini Video Player."""
        if not self.preview_cap or not self.preview_cap.isOpened():
            self.preview_label.configure(text="Select a video to enable preview", image=None)
            return
        if not self.preview_running.is_set():
            self.preview_running.set()
            self.play_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.playback_thread = threading.Thread(target=self.playback_loop)
            self.playback_thread.start()
    
    def stop_preview_playback(self):
        """Stop the Mini Video Player."""
        if self.preview_running.is_set():
            self.preview_running.clear()
            if self.playback_thread:
                self.playback_thread.join()
            self.play_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.preview_label.configure(text="Preview stopped", image=None)
    
    def playback_loop(self):
        """Loop to play preview video."""
        fps = 20  # Target playback FPS (2x speed for 10 FPS source)
        delay = 1 / fps
        while self.preview_running.is_set():
            ret, frame = self.preview_cap.read()
            if not ret:
                self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
                continue
            normalized = normalize_frame(frame, self.clip_limit, self.saturation_multiplier)
            if normalized is not None:
                if is_white_or_black_frame(normalized, self.white_threshold, self.black_threshold):
                    self.preview_label.configure(text="Frame filtered out", image=None)
                else:
                    frame_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    self.preview_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(200, 150))
                    self.preview_label.configure(text="", image=self.preview_image)
            time.sleep(delay)
    
    def on_settings_close(self):
        """Handle settings window close."""
        self.stop_preview_playback()
        if self.preview_cap:
            self.preview_cap.release()
        self.settings_window.destroy()
    
    def save_preset(self):
        """Save current settings as a preset."""
        preset_name = self.preset_name_entry.get().strip()
        if not preset_name:
            messagebox.showwarning("Warning", "Enter a preset name.")
            return
        if preset_name in self.presets:
            messagebox.showwarning("Warning", "Preset name exists.")
            return
        
        self.presets[preset_name] = {
            "motion_threshold": self.motion_threshold,
            "white_threshold": self.white_threshold,
            "black_threshold": self.black_threshold,
            "clip_limit": self.clip_limit,
            "saturation_multiplier": self.saturation_multiplier
        }
        with open("presets.json", "w") as f:
            json.dump(self.presets, f)
        self.preset_combobox.configure(values=list(self.presets.keys()))
        messagebox.showinfo("Info", f"Preset '{preset_name}' saved.")
    
    def load_preset(self):
        """Load selected preset."""
        preset = self.preset_combobox.get()
        if preset in self.presets:
            settings = self.presets[preset]
            self.motion_slider.set(settings["motion_threshold"])
            self.white_slider.set(settings["white_threshold"])
            self.black_slider.set(settings["black_threshold"])
            self.clip_slider.set(settings["clip_limit"])
            self.saturation_slider.set(settings["saturation_multiplier"])
            self.update_settings(0)
            messagebox.showinfo("Info", f"Preset '{preset}' loaded.")
    
    def save_settings(self):
        """Save settings to instance and file."""
        self.motion_threshold = int(self.motion_slider.get())
        self.white_threshold = int(self.white_slider.get())
        self.black_threshold = int(self.black_slider.get())
        self.clip_limit = float(self.clip_slider.get())
        self.saturation_multiplier = float(self.saturation_slider.get())
        
        settings = {
            "motion_threshold": self.motion_threshold,
            "white_threshold": self.white_threshold,
            "black_threshold": self.black_threshold,
            "clip_limit": self.clip_limit,
            "saturation_multiplier": self.saturation_multiplier
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        self.stop_preview_playback()
        if self.preview_cap:
            self.preview_cap.release()
        self.settings_window.destroy()
    
    def reset_to_default(self):
        """Reset settings to defaults."""
        self.motion_slider.set(3000)
        self.white_slider.set(200)
        self.black_slider.set(50)
        self.clip_slider.set(1.0)
        self.saturation_slider.set(1.1)
        self.update_settings(0)
    
    def browse_files(self):
        """Handle multiple file selection."""
        self.input_files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.input_files:
            self.label.configure(text=f"Selected: {len(self.input_files)} file(s)")
            self.start_button.configure(state="normal")
    
    def start_processing(self):
        """Initiate video processing."""
        if not self.input_files:
            messagebox.showwarning("Warning", "No files selected.")
            return
        if not self.generate_60s.get() and not self.generate_12min.get():
            messagebox.showwarning("Warning", "Select at least one video to generate.")
            return
        self.output_text.delete("0.0", tk.END)
        self.output_text.insert("0.0", "Processing...\n")
        self.switch_60s.configure(state="disabled")
        self.switch_12min.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.cancel_event.clear()
        self.start_time = time.time()
        threading.Thread(target=self.process_video_thread).start()
    
    def process_video_thread(self):
        """Process videos in background."""
        selected_videos = []
        if self.generate_60s.get():
            selected_videos.append(("Generate 60s Video", 60))
        if self.generate_12min.get():
            selected_videos.append(("Generate 12min Video", 720))
        
        output_format = self.output_format_var.get()
        for input_file in self.input_files:
            base, _ = os.path.splitext(input_file)
            output_files = {}
            
            for task_name, duration in selected_videos:
                if self.cancel_event.is_set():
                    self.queue.put(("canceled", "User Cancellation"))
                    break
                output_file = f"{base}_{task_name.split()[1]}.{output_format}"
                self.queue.put(("task_start", f"{task_name} - {os.path.basename(input_file)}", 0))
                
                def progress_callback(progress, current, total, remaining):
                    self.queue.put(("progress", progress, (current / total) * 100, remaining))
                
                error = process_video_multi_pass(
                    input_file, output_file, duration,
                    motion_threshold=self.motion_threshold,
                    sample_interval=5,
                    white_threshold=self.white_threshold,
                    black_threshold=self.black_threshold,
                    clip_limit=self.clip_limit,
                    saturation_multiplier=self.saturation_multiplier,
                    output_format=output_format,
                    progress_callback=progress_callback,
                    cancel_event=self.cancel_event
                )
                
                if error:
                    self.queue.put(("canceled", error))
                    break
                else:
                    output_files[task_name] = output_file
                    self.queue.put(("complete", output_files, time.time() - self.start_time))
    
    def cancel_processing(self):
        """Cancel ongoing processing."""
        self.cancel_event.set()
    
    def process_queue(self):
        """Update GUI from queue."""
        try:
            while True:
                message = self.queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)
    
    def handle_message(self, message):
        """Handle queue messages."""
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
            output_files, elapsed = message[1:]
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes} min {seconds} sec"
            self.output_text.delete("0.0", tk.END)
            for task, file in output_files.items():
                self.output_text.insert(tk.END, f"{task}: {file}\n")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Complete in {time_str}")
            self.reset_ui()
        elif message[0] == "canceled":
            reason = message[1]
            elapsed = time.time() - self.start_time if self.start_time else 0
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes} min {seconds} sec"
            self.output_text.delete("0.0", tk.END)
            self.output_text.insert("0.0", f"Canceled: {reason}\n")
            self.current_task_label.configure(text="Current Task: N/A")
            self.time_label.configure(text=f"Process Canceled in {time_str}")
            self.reset_ui()
    
    def reset_ui(self):
        """Reset UI state."""
        self.switch_60s.configure(state="normal")
        self.switch_12min.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.start_button.configure(state="normal" if self.input_files else "disabled")
        self.cancel_button.configure(state="disabled")

### Entry Point

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()