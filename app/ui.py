import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from utils import ToolTip, load_settings, save_settings, load_presets, save_presets, system_check, log_session, SETTINGS_FILE, check_network_stability
from video_processing import process_single_video, debug_get_selected_indices, debug_normalize_frame, debug_generate_output_video
from youtube_upload import start_upload, debug_upload_to_youtube, get_youtube_client
import psutil
try:
    import pygame
except ImportError:
    pygame = None
try:
    from tkinterdnd2 import DND_FILES
except ImportError:
    DND_FILES = None

class VideoProcessorApp:
    VERSION = "10.1.0"

    def __init__(self, root):
        self.root = root
        self.root.title("Bird Box Video Processor")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        try:
            system_check()
        except ValueError as e:
            messagebox.showerror("System Check Failed", str(e))
            self.root.quit()

        self.settings = load_settings()
        self.presets = load_presets()
        self.default_settings = {
            'motion_threshold': 5000,
            'white_threshold': 240,
            'black_threshold': 10,
            'clip_limit': 2.0,
            'saturation_boost': 1.2,
            'music_volume': 1.0,
            'flow_threshold': 0.5,
            'fade_in_out': 1.0,
            'output_format': 'mp4',
            'output_dir': os.getcwd(),
            'resolution': '1920x1080',
            'watermark': '',
            'custom_ffmpeg_args': '',
            'batch_size': 100,
            'workers': psutil.cpu_count() - 1,
            'music_paths': {},
            'update_channel': 'Stable'
        }
        self.settings = {**self.default_settings, **self.settings}

        if psutil.virtual_memory().total / (1024 ** 3) < 8:
            self.settings['batch_size'] = 50

        self.files = []
        self.progress_queues = {}
        self.cancel_events = {}
        self.pause_events = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.analytics_data = []
        self.total_progress = 0

        self.setup_ui()
        self.auto_save_settings()
        log_session(f"Started version {self.VERSION} at {time.time()}")

    def setup_ui(self):
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill='both', expand=True)

        tabs = ["Main", "Settings", "Music", "Advanced", "Help", "Debug"]
        for tab in tabs:
            self.tabview.add(tab)

        self.setup_main_tab()
        self.setup_settings_tab()
        self.setup_music_tab()
        self.setup_advanced_tab()
        self.setup_help_tab()
        self.setup_debug_tab()

        self.theme_switch = ctk.CTkSwitch(self.root, text="Light Mode", command=self.toggle_theme)
        self.theme_switch.pack(side='bottom')

        self.total_progress_bar = ctk.CTkProgressBar(self.root)
        self.total_progress_bar.pack(side='bottom', fill='x')

    def toggle_theme(self):
        mode = "Light" if self.theme_switch.get() else "Dark"
        ctk.set_appearance_mode(mode)

    def setup_main_tab(self):
        tab = self.tabview.tab("Main")
        self.file_list = ctk.CTkScrollableFrame(tab)
        self.file_list.pack(fill='both', expand=True)

        browse_btn = ctk.CTkButton(tab, text="Browse Files", command=self.browse_files)
        browse_btn.pack()

        start_btn = ctk.CTkButton(tab, text="Start Processing", command=self.start_processing)
        start_btn.pack()

        pause_btn = ctk.CTkButton(tab, text="Pause", command=self.pause_processing)
        pause_btn.pack()

        cancel_btn = ctk.CTkButton(tab, text="Cancel", command=self.cancel_processing)
        cancel_btn.pack()

        self.output_frame = ctk.CTkFrame(tab)
        self.output_frame.pack(fill='both', expand=True)

        if hasattr(self.root, 'drop_target_register') and DND_FILES is not None:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.drop_files)

        self.root.bind("<Control-o>", lambda e: self.browse_files())
        self.root.bind("<Control-s>", lambda e: self.start_processing())
        self.root.bind("<Control-c>", lambda e: self.cancel_processing())

        self.undo_btn = ctk.CTkButton(tab, text="Undo Last File", command=self.undo_file)
        self.undo_btn.pack()

        self.preview_label = ctk.CTkLabel(tab, text="Preview")
        self.preview_label.pack()
        self.preview_label.bind("<Double-1>", self.toggle_fullscreen_preview)

    def toggle_fullscreen_preview(self, event):
        pass

    def browse_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        self.add_files(files)

    def drop_files(self, event):
        files = self.root.tk.splitlist(event.data)
        self.add_files(files)

    def add_files(self, files):
        for file in files:
            if file not in self.files:
                self.files.append(file)
                frame = ctk.CTkFrame(self.file_list)
                label = ctk.CTkLabel(frame, text=os.path.basename(file))
                label.pack(side='left')
                progress_bar = ctk.CTkProgressBar(frame)
                progress_bar.pack(side='left', fill='x', expand=True)
                status_label = ctk.CTkLabel(frame, text="Pending")
                status_label.pack(side='left')
                cancel_btn = ctk.CTkButton(frame, text="Cancel", command=lambda f=file: self.cancel_file(f))
                cancel_btn.pack(side='right')
                frame.pack(fill='x')
                self.progress_queues[file] = (progress_bar, status_label)
                self.cancel_events[file] = multiprocessing.Event()
                self.pause_events[file] = multiprocessing.Event()

    def undo_file(self):
        if self.files:
            last_file = self.files.pop()
            pass

    def start_processing(self):
        total_tasks = len(self.files)
        self.total_progress = 0
        for file in self.files:
            self.executor.submit(self.process_video_thread, file)

    def process_video_thread(self, file):
        q = queue.Queue()

        def progress_callback(progress):
            q.put(progress)

        try:
            output, stats = process_single_video(file, self.settings, progress_callback, self.cancel_events[file], self.pause_events[file])
            self.analytics_data.append(stats)
            self.show_output(file, output)
        except Exception as e:
            messagebox.showerror("Error", str(e))

        def update_progress():
            while not q.empty():
                progress = q.get()
                bar, status = self.progress_queues[file]
                bar.set(progress / 100)
                status.configure(text=f"{progress}%")
                self.total_progress += (progress / total_tasks)
                self.total_progress_bar.set(self.total_progress / 100)
            self.root.after(100, update_progress)

        update_progress()

    def pause_processing(self):
        for event in self.pause_events.values():
            event.set()

    def cancel_processing(self):
        for event in self.cancel_events.values():
            event.set()

    def cancel_file(self, file):
        self.cancel_events[file].set()

    def show_output(self, input_file, output_path):
        frame = ctk.CTkFrame(self.output_frame)
        label = ctk.CTkLabel(frame, text=os.path.basename(output_path), fg_color="blue", cursor="hand2")
        label.pack(side='left')
        label.bind("<Button-1>", lambda e: os.startfile(output_path))
        upload_btn = ctk.CTkButton(frame, text="Upload to YouTube", command=lambda: self.upload_output(output_path))
        upload_btn.pack(side='right')
        frame.pack(fill='x')

    def upload_output(self, path):
        threading.Thread(target=start_upload, args=(path, "Title", "Desc", ["tags"],)).start()

    def setup_settings_tab(self):
        tab = self.tabview.tab("Settings")

        sliders = {
            'motion_threshold': (1000, 10000),
            'white_threshold': (0, 255),
            'black_threshold': (0, 255),
            'clip_limit': (0.1, 5.0),
            'saturation_boost': (0.5, 2.0),
            'flow_threshold': (0.1, 2.0),
            'music_volume': (0.1, 2.0),
            'fade_in_out': (0, 5)
        }

        for key, (min_val, max_val) in sliders.items():
            label = ctk.CTkLabel(tab, text=key.replace('_', ' ').title())
            label.pack()
            slider = ctk.CTkSlider(tab, from_=min_val, to=max_val, command=lambda v, k=key: self.update_setting(k, v))
            slider.set(self.settings[key])
            slider.pack()
            ToolTip(slider, f"Adjust {key}")

        reset_btn = ctk.CTkButton(tab, text="Reset to Default", command=self.reset_settings)
        reset_btn.pack()

        self.update_preview()

    def update_setting(self, key, value):
        self.settings[key] = value
        save_settings(self.settings)
        self.update_preview()

    def reset_settings(self):
        self.settings = self.default_settings.copy()
        save_settings(self.settings)
        pass

    def update_preview(self):
        pass

    def setup_music_tab(self):
        tab = self.tabview.tab("Music")

        durations = ['default', '60', '720', '3600']
        for dur in durations:
            label = ctk.CTkLabel(tab, text=f"Music for {dur}s")
            label.pack()
            btn = ctk.CTkButton(tab, text="Select File", command=lambda d=dur: self.select_music(d))
            btn.pack()
            preview_btn = ctk.CTkButton(tab, text="Preview", command=lambda d=dur: self.preview_music(d))
            preview_btn.pack()

    def select_music(self, dur):
        file = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav")])
        if file:
            self.settings['music_paths'][dur] = file
            save_settings(self.settings)

    def preview_music(self, dur):
        path = self.settings['music_paths'].get(dur)
        if path:
            if pygame:
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            else:
                os.startfile(path)

    def setup_advanced_tab(self):
        tab = self.tabview.tab("Advanced")

        format_label = ctk.CTkLabel(tab, text="Output Format")
        format_label.pack()
        self.format_combo = ctk.CTkComboBox(tab, values=["mp4", "mkv", "avi"], command=lambda v: self.update_setting('output_format', v))
        self.format_combo.set(self.settings['output_format'])
        self.format_combo.pack()

        dir_btn = ctk.CTkButton(tab, text="Select Output Dir", command=self.select_output_dir)
        dir_btn.pack()

        res_label = ctk.CTkLabel(tab, text="Resolution (WxH)")
        res_label.pack()
        res_entry = ctk.CTkEntry(tab)
        res_entry.insert(0, self.settings['resolution'])
        res_entry.bind("<FocusOut>", lambda e: self.validate_resolution(res_entry.get()))
        res_entry.pack()

        wm_label = ctk.CTkLabel(tab, text="Watermark Text")
        wm_label.pack()
        wm_entry = ctk.CTkEntry(tab)
        wm_entry.insert(0, self.settings['watermark'])
        wm_entry.bind("<FocusOut>", lambda e: self.update_setting('watermark', wm_entry.get()))
        wm_entry.pack()

        ffmpeg_label = ctk.CTkLabel(tab, text="Custom FFmpeg Args")
        ffmpeg_label.pack()
        ffmpeg_entry = ctk.CTkEntry(tab)
        ffmpeg_entry.insert(0, self.settings['custom_ffmpeg_args'])
        ffmpeg_entry.bind("<FocusOut>", lambda e: self.update_setting('custom_ffmpeg_args', ffmpeg_entry.get()))
        ffmpeg_entry.pack()

        for key in ['batch_size', 'workers']:
            label = ctk.CTkLabel(tab, text=key.title())
            label.pack()
            entry = ctk.CTkEntry(tab)
            entry.insert(0, str(self.settings[key]))
            entry.bind("<FocusOut>", lambda e, k=key: self.update_setting(k, int(entry.get())))
            entry.pack()

        preset_combo = ctk.CTkComboBox(tab, values=list(self.presets.keys()), command=self.load_preset)
        preset_combo.pack()
        save_preset_btn = ctk.CTkButton(tab, text="Save Preset", command=self.save_preset)
        save_preset_btn.pack()
        delete_preset_btn = ctk.CTkButton(tab, text="Delete Preset", command=self.delete_preset)
        delete_preset_btn.pack()

        update_label = ctk.CTkLabel(tab, text="Update Channel")
        update_label.pack()
        update_combo = ctk.CTkComboBox(tab, values=["Stable", "Beta"], command=lambda v: self.update_setting('update_channel', v))
        update_combo.set(self.settings['update_channel'])
        update_combo.pack()

        export_btn = ctk.CTkButton(tab, text="Export Analytics to CSV", command=self.export_analytics_csv)
        export_btn.pack()

    def select_output_dir(self):
        dir = filedialog.askdirectory()
        if dir:
            self.settings['output_dir'] = dir
            save_settings(self.settings)

    def validate_resolution(self, res_str):
        try:
            w, h = map(int, res_str.split('x'))
            self.settings['resolution'] = res_str
            save_settings(self.settings)
        except ValueError:
            messagebox.showerror("Invalid", "Resolution must be WxH")

    def save_preset(self):
        name = tk.simpledialog.askstring("Preset Name", "Enter name")
        if name:
            self.presets[name] = self.settings.copy()
            save_presets(self.presets)

    def load_preset(self, name):
        if name in self.presets:
            self.settings = self.presets[name].copy()
            save_settings(self.settings)

    def delete_preset(self):
        name = self.preset_combo.get()
        if name in self.presets:
            del self.presets[name]
            save_presets(self.presets)

    def export_analytics_csv(self):
        with open("analytics.csv", "w") as f:
            pass

    def setup_help_tab(self):
        tab = self.tabview.tab("Help")
        text = ctk.CTkTextbox(tab)
        text.insert("0.0", "Usage...\nShortcuts...\nTips...")
        text.pack(fill='both', expand=True)

        search_entry = ctk.CTkEntry(tab)
        search_entry.pack()
        search_entry.bind("<KeyRelease>", lambda e: self.search_faq(text, search_entry.get()))

    def search_faq(self, textbox, query):
        pass

    def setup_debug_tab(self):
        tab = self.tabview.tab("Debug")

        debug_indices_btn = ctk.CTkButton(tab, text="Debug Selected Indices", command=lambda: debug_get_selected_indices("test.mp4"))
        debug_indices_btn.pack()

        debug_norm_btn = ctk.CTkButton(tab, text="Debug Normalize Frame", command=lambda: debug_normalize_frame(np.zeros((100,100,3))))
        debug_norm_btn.pack()

        debug_gen_btn = ctk.CTkButton(tab, text="Debug Generate Video", command=debug_generate_output_video)
        debug_gen_btn.pack()

        debug_upload_btn = ctk.CTkButton(tab, text="Debug Upload", command=lambda: debug_upload_to_youtube("test.mp4"))
        debug_upload_btn.pack()

        self.log_text = ctk.CTkTextbox(tab)
        self.log_text.pack(fill='both', expand=True)
        self.update_logs()

        test_net_btn = ctk.CTkButton(tab, text="Test Network", command=check_network_stability)
        test_net_btn.pack()

        test_auth_btn = ctk.CTkButton(tab, text="Test YouTube Auth", command=get_youtube_client)
        test_auth_btn.pack()

    def update_logs(self):
        self.root.after(1000, self.update_logs)

    def show_analytics_dashboard(self):
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()
        summary_tab = ctk.CTkTabview(self.root).add("Summary")

    def auto_save_settings(self):
        pass