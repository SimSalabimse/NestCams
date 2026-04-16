"""
Bird Box Video Processor  v12.0  —  GUI
═══════════════════════════════════════════════════════════════════════════════

New in v12.0 (vs v11):
  • ROI picker: draw rectangle on first frame preview, save to config
  • Motion tab: sensitivity, frame-skip, ROI, MOG2, contour area, flicker ratio
  • Colour/Smoothing tab: deflicker strength, contrast, brightness, saturation,
    motion blur frames (auto or manual), denoise toggle
  • Full settings load/save/reset (all sliders, all text fields, all paths)
  • Full preset save/load/delete (any named configuration snapshot)
  • Rich Analytics tab: per-file table + session log viewer
  • Improved Help tab with pipeline order explanation
  • All logging goes through utils.py structured logger
"""

import json
import os
import queue
import threading
import time
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import requests
import schedule
import tkinter as tk
import customtkinter as ctk
from packaging import version
from PIL import Image, ImageDraw
from tkinter import filedialog, messagebox

from motion_detector import MotionDetector
from video_processor  import VideoProcessor
from youtube_upload   import start_upload
from utils            import log_session, ToolTip

try:
    from tkinterdnd2 import DND_FILES
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

import logging
logger = logging.getLogger(__name__)

VERSION       = "12.0.0"
GITHUB_REPO   = "SimSalabimse/NestCams"
SETTINGS_FILE = "settings.json"
PRESETS_FILE  = "presets.json"

QUALITY_LABELS = ["Low (faster)", "Medium", "High", "Maximum (slower)"]

DEFAULTS: dict = {
    # Motion
    "sensitivity":        5,
    "white_threshold":  225,
    "black_threshold":   35,
    "segment_padding":  0.3,
    "frame_skip":         2,
    "merge_gap":         0.8,
    "min_motion_duration": 0.4,
    "roi":               None,
    "use_mog2":         False,
    "mog2_learning_rate": 0.005,
    # Colour / Smoothing
    "deflicker_size":     5,
    "contrast":           1.0,
    "brightness":         0.0,
    "saturation":         1.0,
    "motion_blur_frames": -1,   # -1 = auto
    "denoise":           True,
    # Output
    "quality":            2,
    "use_gpu":           True,
    "watermark_text":    None,
    "custom_ffmpeg_args": None,
    # Music
    "music_volume":       0.5,
    "music_paths": {"default": None, "60": None, "720": None, "3600": None},
    # General
    "output_dir":        None,
    "update_channel":    "Stable",
}


# ═════════════════════════════════════════════════════════════════════════════
class VideoProcessorApp:
# ═════════════════════════════════════════════════════════════════════════════

    def __init__(self, root: tk.Tk, has_dnd: bool = False) -> None:
        self.root    = root
        self.has_dnd = has_dnd

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        self.root.title(f"Bird Box Video Processor v{VERSION}")
        self.root.geometry("1100x820")
        self.root.resizable(True, True)
        log_session(f"App started v{VERSION}")
        logger.info(f"App started v{VERSION}")

        # Fonts — created AFTER root window exists
        self.F_BODY = ctk.CTkFont(size=13)
        self.F_HEAD = ctk.CTkFont(size=14, weight="bold")
        self.F_BTN  = ctk.CTkFont(size=13)
        self.F_PROG = ctk.CTkFont(size=12)
        self.F_MONO = ctk.CTkFont(family="Courier New", size=11)

        # Runtime state
        self.input_files:    List[str]  = []
        self.progress_rows:  Dict       = {}
        self.analytics_data: List[dict] = []
        self.presets:        Dict       = {}
        self.queue:          queue.Queue = queue.Queue()
        self.preview_queue:  queue.Queue = queue.Queue(maxsize=5)
        self.preview_cap:    Optional[cv2.VideoCapture] = None
        self.preview_thread: Optional[threading.Thread] = None
        self.preview_running = False
        self.preview_image   = None
        self._processing     = False
        self._cancel_flag    = threading.Event()
        self._pause_flag     = threading.Event()
        self._paused         = False
        self._active_proc:   Optional[VideoProcessor] = None
        self._proc_start:    float = 0.0
        self._roi_drawing    = False   # True while user is drawing ROI
        self._roi_start: Optional[Tuple[int,int]] = None

        # Mutable settings state (mirrors DEFAULTS)
        self._music_paths: Dict[str, Optional[str]] = {
            k: None for k in ("default", "60", "720", "3600")
        }
        self._output_dir: Optional[str] = None
        self._roi:         Optional[dict] = None

        self.blank_img = ctk.CTkImage(
            light_image=Image.new("RGB", (310, 196), (18, 18, 18)),
            dark_image=Image.new("RGB", (310, 196), (18, 18, 18)),
            size=(310, 196),
        )

        self._check_system()
        self._build_tabs()
        self._load_settings()
        self._load_presets()

        if has_dnd:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

        self.root.bind("<Control-o>", lambda _: self.browse_files())
        self.root.bind("<Control-s>", lambda _: self._kb_start())
        self.root.bind("<Control-c>", lambda _: self._kb_cancel())

        self.root.after(50, self._process_queue)
        self.root.after(33, self._update_preview)
        threading.Thread(target=self._check_updates, daemon=True).start()

    # ── System ────────────────────────────────────────────────────────────
    def _check_system(self) -> None:
        self.opencl_ok = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
        self.nvenc_ok  = False
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5,
            )
            self.nvenc_ok = "h264_nvenc" in r.stdout
        except Exception:
            pass
        self.cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        log_session(
            f"System: opencl={self.opencl_ok} nvenc={self.nvenc_ok} "
            f"threads={self.cpu_threads} ram={ram_gb:.1f}GB"
        )
        logger.info(
            f"System check: opencl={self.opencl_ok} "
            f"nvenc={self.nvenc_ok} "
            f"threads={self.cpu_threads} "
            f"ram={ram_gb:.1f}GB"
        )

    # ── Tab scaffold ──────────────────────────────────────────────────────
    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self.root)
        self.tabs.pack(pady=10, padx=10, fill="both", expand=True)
        for t in ("Main", "Motion", "Colour", "Music", "Advanced",
                  "Analytics", "Help"):
            self.tabs.add(t)
        self._build_main_tab()
        self._build_motion_tab()
        self._build_colour_tab()
        self._build_music_tab()
        self._build_advanced_tab()
        self._build_analytics_tab()
        self._build_help_tab()

    # ══════════════════════════════════════════════════════════════════════
    # MAIN TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_main_tab(self) -> None:
        tab = self.tabs.tab("Main")

        # GPU banner
        parts = (["OpenCL motion"] if self.opencl_ok else []) + \
                (["NVENC encode"]   if self.nvenc_ok  else [])
        btxt = ("✅  GPU active — " + ", ".join(parts)) if parts else \
               "ℹ️  CPU mode — update GPU drivers for OpenCL / NVENC"
        bcol = "#16a34a" if parts else "#d97706"
        ctk.CTkLabel(tab, text=btxt, fg_color=bcol, corner_radius=6,
                     text_color="white", padx=10,
                     font=self.F_BODY).pack(fill="x", padx=10, pady=(8, 4))

        top = ctk.CTkFrame(tab)
        top.pack(fill="x", padx=10, pady=4)

        # ── Left column ────────────────────────────────────────────────────
        left = ctk.CTkFrame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        hint = " · Drag & Drop" if self.has_dnd else ""
        self.file_label = ctk.CTkLabel(
            left, text=f"No files selected{hint}", font=self.F_BODY)
        self.file_label.pack(pady=6)

        self.browse_button = ctk.CTkButton(
            left, text="Browse Files  (Ctrl+O)",
            command=self.browse_files, font=self.F_BTN)
        self.browse_button.pack(pady=4)

        dur = ctk.CTkFrame(left); dur.pack(pady=6)
        self.gen_60s = tk.BooleanVar(value=True)
        self.gen_12m = tk.BooleanVar(value=True)
        self.gen_1h  = tk.BooleanVar(value=True)
        self.sw_60s = ctk.CTkSwitch(dur, text="60s",   variable=self.gen_60s, font=self.F_BODY)
        self.sw_12m = ctk.CTkSwitch(dur, text="12min", variable=self.gen_12m, font=self.F_BODY)
        self.sw_1h  = ctk.CTkSwitch(dur, text="1hr",   variable=self.gen_1h,  font=self.F_BODY)
        for sw in (self.sw_60s, self.sw_12m, self.sw_1h):
            sw.pack(side="left", padx=10)
        ToolTip(self.sw_60s, "60-second vertical Short (auto-rotated 90°)")
        ToolTip(self.sw_12m, "12-minute highlights reel")
        ToolTip(self.sw_1h,  "1-hour reel")

        cf = ctk.CTkFrame(left); cf.pack(pady=4)
        ctk.CTkLabel(cf, text="Custom (s):", font=self.F_BODY).pack(side="left", padx=4)
        self.custom_duration = ctk.CTkEntry(
            cf, width=90, placeholder_text="e.g. 300", font=self.F_BODY)
        self.custom_duration.pack(side="left")

        bf = ctk.CTkFrame(left); bf.pack(pady=8)
        self.start_button = ctk.CTkButton(
            bf, text="▶  Start Processing", command=self.start_processing,
            state="disabled", fg_color="#16a34a", hover_color="#15803d",
            font=self.F_HEAD, height=42)
        self.start_button.pack(side="left", padx=6)
        ToolTip(self.start_button, "Ctrl+S")

        self.pause_button = ctk.CTkButton(
            bf, text="⏸  Pause", command=self._toggle_pause,
            state="disabled", font=self.F_BTN, height=42)
        self.pause_button.pack(side="left", padx=4)

        self.cancel_button = ctk.CTkButton(
            bf, text="✕  Cancel", command=self._cancel_processing,
            state="disabled", fg_color="#dc2626", hover_color="#b91c1c",
            font=self.F_BTN, height=42)
        self.cancel_button.pack(side="left", padx=4)
        ToolTip(self.cancel_button, "Cancel immediately (Ctrl+C)")

        pg = ctk.CTkFrame(left); pg.pack(fill="x", pady=(8, 0), padx=4)
        self.overall_bar = ctk.CTkProgressBar(pg, height=18)
        self.overall_bar.pack(fill="x", padx=4, pady=(4, 2))
        self.overall_bar.set(0)
        self.overall_label = ctk.CTkLabel(
            pg, text="", font=self.F_PROG, text_color="gray70")
        self.overall_label.pack(anchor="w", padx=6)

        # ── Right column: preview ─────────────────────────────────────────
        right = ctk.CTkFrame(top); right.pack(side="right", padx=(6, 0))

        self.preview_label = ctk.CTkLabel(right, image=self.blank_img, text="")
        self.preview_label.pack(padx=6, pady=4)

        self.preview_slider = ctk.CTkSlider(
            right, from_=0, to=1, command=self._seek_preview)
        self.preview_slider.pack(fill="x", padx=6)
        self.preview_slider.set(0)

        prow = ctk.CTkFrame(right); prow.pack(pady=4)
        self.preview_button = ctk.CTkButton(
            prow, text="▶  Preview", command=self._toggle_preview,
            width=130, state="disabled", font=self.F_BTN)
        self.preview_button.pack(side="left", padx=2)
        self.roi_button = ctk.CTkButton(
            prow, text="📐 Set ROI", command=self._start_roi_picker,
            width=110, state="disabled", font=self.F_BTN,
            fg_color="#7c3aed", hover_color="#6d28d9")
        self.roi_button.pack(side="left", padx=2)
        ToolTip(self.roi_button,
                "Click then drag on the preview to define a Region of Interest.\n"
                "Motion outside this rectangle will be ignored.")
        self.roi_status = ctk.CTkLabel(
            right, text="ROI: off", font=self.F_PROG, text_color="gray60")
        self.roi_status.pack()

        # Progress + output
        ctk.CTkLabel(tab, text="Per-file progress:",
                     font=self.F_BODY).pack(anchor="w", padx=14, pady=(6, 0))
        self.progress_frame = ctk.CTkScrollableFrame(tab, height=175)
        self.progress_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(tab, text="Output files:",
                     font=self.F_BODY).pack(anchor="w", padx=14, pady=(4, 0))
        self.output_frame = ctk.CTkScrollableFrame(tab, height=105)
        self.output_frame.pack(fill="x", padx=10, pady=2)

    # ══════════════════════════════════════════════════════════════════════
    # MOTION TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_motion_tab(self) -> None:
        tab = self.tabs.tab("Motion")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        def _row(label, lo, hi, steps, default, fmt, tip=""):
            ctk.CTkLabel(f, text=label, font=self.F_BODY).pack(pady=(8, 0), anchor="w", padx=10)
            var = tk.DoubleVar(value=default)
            sl  = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps, variable=var)
            sl.pack(pady=2, fill="x", padx=30)
            lbl = ctk.CTkLabel(f, text=fmt.format(default), font=self.F_PROG)
            lbl.pack()
            var.trace_add("write", lambda *_: lbl.configure(text=fmt.format(var.get())))
            if tip: ToolTip(sl, tip)
            return sl, var

        self.sens_sl, self.sens_var = _row(
            "Motion Sensitivity  (1 = very sensitive · 10 = large movements only)",
            1, 10, 9, DEFAULTS["sensitivity"], "{:.0f}",
            "Low values catch subtle nestling movements.\n"
            "Raise if wind/shadows cause false detections.")

        self.skip_sl, self.skip_var = _row(
            "Frame Skip  (analyse every Nth frame — fine pass always uses 1)",
            1, 8, 7, DEFAULTS["frame_skip"], "{:.0f}",
            "Higher = faster scan.  2 is a good balance for 30 fps cameras.")

        self.pad_sl, self.pad_var = _row(
            "Segment Padding  (seconds of context kept before/after each event)",
            0.0, 5.0, 50, DEFAULTS["segment_padding"], "{:.2f}s",
            "0.2–0.5 s removes most dead footage.\n"
            "Increase only if events are getting cut off.")

        self.merge_sl, self.merge_var = _row(
            "Merge Gap  (gaps shorter than this are merged into one segment)",
            0.1, 5.0, 49, DEFAULTS["merge_gap"], "{:.1f}s",
            "Prevents fragmentation of a single visit into many tiny segments.")

        self.white_sl, self.white_var = _row(
            "White Frame Threshold  (skip frames brighter than this average)",
            100, 255, 155, DEFAULTS["white_threshold"], "{:.0f}",
            "Frames with mean brightness above this are skipped (overexposed/IR).")

        self.black_sl, self.black_var = _row(
            "Black Frame Threshold  (skip frames darker than this average)",
            0, 80, 80, DEFAULTS["black_threshold"], "{:.0f}",
            "Frames with mean brightness below this are skipped (night drop-outs).")

        # MOG2
        sep = ctk.CTkFrame(f, height=2, fg_color="gray30"); sep.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f, text="Background Subtraction (MOG2)",
                     font=self.F_HEAD).pack(pady=(4, 0))
        self.mog2_var = tk.BooleanVar(value=DEFAULTS["use_mog2"])
        ctk.CTkSwitch(f, text="Enable MOG2 secondary check",
                      variable=self.mog2_var, font=self.F_BODY).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "MOG2 builds a background model and requires both the diff-test\n"
                "AND the background subtractor to agree before marking a frame.\n"
                "Reduces false positives at the cost of ~10% extra processing.")

        self.mog2_sl, self.mog2_var2 = _row(
            "MOG2 Learning Rate  (how quickly the background adapts)",
            0.001, 0.05, 49, DEFAULTS["mog2_learning_rate"], "{:.4f}",
            "Lower = slower adaptation (better for static cameras).\n"
            "0.005 is ideal for a box camera that doesn't move.")

        ctk.CTkButton(f, text="💾  Save Motion Settings",
                      command=self._save_settings, font=self.F_BTN).pack(pady=12)

    # ══════════════════════════════════════════════════════════════════════
    # COLOUR / SMOOTHING TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_colour_tab(self) -> None:
        tab = self.tabs.tab("Colour")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        def _row(label, lo, hi, steps, default, fmt, tip=""):
            ctk.CTkLabel(f, text=label, font=self.F_BODY).pack(pady=(8, 0), anchor="w", padx=10)
            var = tk.DoubleVar(value=default)
            sl  = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps, variable=var)
            sl.pack(pady=2, fill="x", padx=30)
            lbl = ctk.CTkLabel(f, text=fmt.format(default), font=self.F_PROG)
            lbl.pack()
            var.trace_add("write", lambda *_: lbl.configure(text=fmt.format(var.get())))
            if tip: ToolTip(sl, tip)
            return sl, var

        self.deflicker_sl, self.deflicker_var = _row(
            "Deflicker Strength  (sliding window size — 0 = off)",
            0, 15, 15, DEFAULTS["deflicker_size"], "{:.0f}",
            "Corrects per-frame brightness spikes.\n"
            "5 is ideal.  Increase for severe flicker, 0 to disable.")

        self.contrast_sl, self.contrast_var = _row(
            "Contrast  (1.0 = neutral, 1.05–1.15 = slightly punchy)",
            0.8, 1.5, 70, DEFAULTS["contrast"], "{:.2f}",
            "Applied via eq= filter before speedup.")

        self.brightness_sl, self.brightness_var = _row(
            "Brightness  (0 = neutral, positive = brighter)",
            -0.3, 0.3, 60, DEFAULTS["brightness"], "{:.2f}",
            "Useful for cameras with dark default exposure.")

        self.sat_sl, self.sat_var = _row(
            "Saturation  (1.0 = neutral, 1.05–1.15 = warmer colours)",
            0.5, 1.8, 130, DEFAULTS["saturation"], "{:.2f}")

        sep = ctk.CTkFrame(f, height=2, fg_color="gray30"); sep.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f, text="Quality & Motion Blur",
                     font=self.F_HEAD).pack(pady=(4, 0))

        ctk.CTkLabel(f, text="Output Quality", font=self.F_BODY).pack(pady=(8, 0))
        self.quality_var = tk.StringVar(value=QUALITY_LABELS[DEFAULTS["quality"]])
        ctk.CTkOptionMenu(f, variable=self.quality_var,
                         values=QUALITY_LABELS, font=self.F_BTN).pack(pady=4)

        self.denoise_var = tk.BooleanVar(value=DEFAULTS["denoise"])
        ctk.CTkSwitch(f, text="Apply hqdn3d light denoising",
                      variable=self.denoise_var, font=self.F_BODY).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "Removes sensor noise and compression artefacts before encoding.\n"
                "Improves perceived sharpness with negligible blur.")

        self.blur_sl, self.blur_var = _row(
            "Motion Blur Frames  (-1 = auto, 0 = off, 3–8 = manual for 60s Short)",
            -1, 12, 13, DEFAULTS["motion_blur_frames"], "{:.0f}",
            "-1: automatically chosen based on speed factor (recommended).\n"
            "0: disabled.  3–8: cinematic blur for the 60s Short.\n"
            "Only applied to 60-second output.")

        ctk.CTkButton(f, text="💾  Save Colour Settings",
                      command=self._save_settings, font=self.F_BTN).pack(pady=12)

    # ══════════════════════════════════════════════════════════════════════
    # MUSIC TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_music_tab(self) -> None:
        tab = self.tabs.tab("Music")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(f, text="Music Volume", font=self.F_BODY).pack(pady=4)
        self.vol_var = tk.DoubleVar(value=DEFAULTS["music_volume"])
        ctk.CTkSlider(f, from_=0, to=1, number_of_steps=100,
                      variable=self.vol_var).pack(fill="x", padx=30, pady=2)
        self.vol_lbl = ctk.CTkLabel(f, text="50%", font=self.F_PROG)
        self.vol_lbl.pack()
        self.vol_var.trace_add("write",
            lambda *_: self.vol_lbl.configure(text=f"{int(self.vol_var.get()*100)}%"))

        self._music_labels: Dict = {}
        for display, key in [("Default / fallback", "default"),
                              ("60-second Short",    "60"),
                              ("12-minute reel",     "720"),
                              ("1-hour reel",        "3600")]:
            r = ctk.CTkFrame(f); r.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(r, text=display, width=180,
                         anchor="w", font=self.F_BODY).pack(side="left", padx=4)
            lbl = ctk.CTkLabel(r, text="No file", font=self.F_BODY)
            lbl.pack(side="left", padx=4, expand=True, fill="x")
            ctk.CTkButton(r, text="Browse…", width=90,
                          command=lambda k=key, l=lbl: self._pick_music(k, l),
                          font=self.F_BTN).pack(side="right", padx=2)
            ctk.CTkButton(r, text="Clear", width=60,
                          command=lambda k=key, l=lbl: self._clear_music(k, l),
                          font=self.F_BTN).pack(side="right", padx=2)
            self._music_labels[key] = lbl

        ctk.CTkLabel(
            f,
            text="Music is looped automatically to fill the target duration,\n"
                 "then faded out over the last 2 seconds.",
            text_color="gray", font=self.F_PROG,
        ).pack(pady=10)

    # ══════════════════════════════════════════════════════════════════════
    # ADVANCED TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_advanced_tab(self) -> None:
        tab = self.tabs.tab("Advanced")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(f, text="Output folder:", font=self.F_BODY).pack(anchor="w")
        df = ctk.CTkFrame(f); df.pack(fill="x", padx=10, pady=2)
        self.out_dir_label = ctk.CTkLabel(df, text="Same as input", font=self.F_BODY)
        self.out_dir_label.pack(side="left", padx=4, expand=True, fill="x")
        ctk.CTkButton(df, text="Browse…", command=self._select_output_dir,
                      width=90, font=self.F_BTN).pack(side="right", padx=4)

        ctk.CTkLabel(f, text="Watermark text (empty = none):",
                     font=self.F_BODY).pack(pady=(10, 0), anchor="w")
        self.watermark_entry = ctk.CTkEntry(
            f, placeholder_text="© My Channel 2024", font=self.F_BODY)
        self.watermark_entry.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(f, text="Extra FFmpeg arguments (appended to final encode):",
                     font=self.F_BODY).pack(pady=(8, 0), anchor="w")
        self.ffmpeg_entry = ctk.CTkEntry(
            f, placeholder_text="-vf hue=s=0", font=self.F_BODY)
        self.ffmpeg_entry.pack(fill="x", padx=10, pady=2)

        chan = ctk.CTkFrame(f); chan.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(chan, text="Update channel:", font=self.F_BODY).pack(side="left", padx=4)
        self.channel_var = tk.StringVar(value=DEFAULTS["update_channel"])
        ctk.CTkOptionMenu(chan, variable=self.channel_var,
                         values=["Stable", "Beta"], font=self.F_BTN).pack(side="left")

        # YouTube setup guide
        yt = ctk.CTkFrame(f); yt.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(yt, text="YouTube Upload Setup", font=self.F_HEAD).pack(pady=4)
        ctk.CTkLabel(
            yt,
            text=(
                "1. Go to https://console.cloud.google.com\n"
                "2. Create a project → enable 'YouTube Data API v3'\n"
                "3. Credentials → Create OAuth 2.0 (Desktop app) → Download JSON\n"
                "4. Rename to  client_secrets.json  and place in program folder\n"
                "5. First upload opens a browser for one-time authorisation"
            ),
            justify="left", font=self.F_PROG, text_color="gray80",
        ).pack(padx=10, pady=4, anchor="w")
        ctk.CTkButton(
            yt, text="Open Google Cloud Console", font=self.F_BTN,
            command=lambda: webbrowser.open("https://console.cloud.google.com"),
        ).pack(pady=4)

        # Schedule
        sf = ctk.CTkFrame(f); sf.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(sf, text="Schedule daily processing at (HH:MM):", font=self.F_BODY).pack(pady=4)
        self.schedule_entry = ctk.CTkEntry(sf, placeholder_text="23:30", width=120, font=self.F_BODY)
        self.schedule_entry.pack(pady=2)
        ctk.CTkButton(sf, text="Set Schedule", command=self._set_schedule, font=self.F_BTN).pack(pady=4)

        # Presets
        pf = ctk.CTkFrame(f); pf.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(pf, text="Preset Management", font=self.F_HEAD).pack(pady=4)
        self.preset_combo = ctk.CTkComboBox(pf, values=[], width=280, font=self.F_BTN)
        self.preset_combo.pack(pady=4)
        r1 = ctk.CTkFrame(pf); r1.pack(pady=2)
        ctk.CTkButton(r1, text="Load", command=self._load_preset, font=self.F_BTN).pack(side="left", padx=4)
        ctk.CTkButton(r1, text="Delete", command=self._delete_preset,
                      fg_color="gray40", font=self.F_BTN).pack(side="left", padx=4)
        r2 = ctk.CTkFrame(pf); r2.pack(pady=4)
        self.preset_name = ctk.CTkEntry(r2, placeholder_text="Preset name…", width=180, font=self.F_BODY)
        self.preset_name.pack(side="left", padx=4)
        ctk.CTkButton(r2, text="Save as Preset", command=self._save_preset, font=self.F_BTN).pack(side="left", padx=4)

        btns = ctk.CTkFrame(f); btns.pack(pady=12)
        ctk.CTkButton(btns, text="💾  Save All Settings",
                      command=self._save_settings, font=self.F_BTN).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="↺  Reset to Defaults",
                      command=self._reset_defaults, fg_color="gray40",
                      font=self.F_BTN).pack(side="left", padx=6)

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTICS TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_analytics_tab(self) -> None:
        tab = self.tabs.tab("Analytics")
        ctk.CTkLabel(tab, text="Session Analytics",
                     font=self.F_HEAD).pack(pady=(10, 4))
        self.analytics_scroll = ctk.CTkScrollableFrame(tab, height=260)
        self.analytics_scroll.pack(fill="x", padx=10, pady=4)

        ctk.CTkLabel(tab, text="Session Log  (last 200 lines):",
                     font=self.F_BODY).pack(anchor="w", padx=14, pady=(8, 0))
        self.log_viewer = ctk.CTkTextbox(tab, font=self.F_MONO, height=200)
        self.log_viewer.pack(fill="both", expand=True, padx=10, pady=4)
        ctk.CTkButton(tab, text="Refresh Log",
                      command=self._refresh_log_viewer,
                      font=self.F_BTN).pack(pady=4)

    def _refresh_analytics(self) -> None:
        for w in self.analytics_scroll.winfo_children():
            w.destroy()

        if not self.analytics_data:
            ctk.CTkLabel(self.analytics_scroll,
                         text="No data yet — process a video first.",
                         text_color="gray", font=self.F_PROG).pack(pady=10)
            return

        # Header
        hdr = ctk.CTkFrame(self.analytics_scroll)
        hdr.pack(fill="x", pady=1)
        for col, w in [("File", 200), ("Video", 70), ("Motion", 90),
                       ("%", 55), ("Segs", 55), ("Proc", 70), ("Method", 120)]:
            ctk.CTkLabel(hdr, text=col, width=w, anchor="w",
                         font=self.F_BODY, text_color="gray70").pack(side="left", padx=3)

        for d in self.analytics_data:
            dur = d.get("video_duration", 0)
            mot = d.get("motion_duration", 0)
            pct = mot / max(dur, 1) * 100
            r = ctk.CTkFrame(self.analytics_scroll); r.pack(fill="x", pady=1)
            vals = [
                (d.get("file", "?")[:26],          200),
                (f"{dur:.0f}s",                    70),
                (f"{mot:.1f}s",                    90),
                (f"{pct:.1f}%",                    55),
                (str(d.get("motion_segments", 0)), 55),
                (f"{d.get('processing_time',0):.1f}s", 70),
                (d.get("detection_method", "—"),   120),
            ]
            for txt, w in vals:
                ctk.CTkLabel(r, text=txt, width=w, anchor="w",
                             font=self.F_PROG).pack(side="left", padx=3)

    def _refresh_log_viewer(self) -> None:
        """Show last 200 lines of the current session log."""
        import glob
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
        files   = sorted(glob.glob(os.path.join(log_dir, "session_*.txt")))
        if not files:
            return
        with open(files[-1], encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        self.log_viewer.configure(state="normal")
        self.log_viewer.delete("1.0", "end")
        self.log_viewer.insert("end", "".join(lines[-200:]))
        self.log_viewer.configure(state="disabled")
        self.log_viewer.see("end")

    # ══════════════════════════════════════════════════════════════════════
    # HELP TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_help_tab(self) -> None:
        tab = self.tabs.tab("Help")
        box = ctk.CTkTextbox(tab, wrap="word", font=self.F_BODY)
        box.pack(fill="both", expand=True, padx=10, pady=10)
        box.insert("end", f"""\
Bird Box Video Processor  v{VERSION}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PIPELINE ORDER  (why it is optimal)
────────────────────────────────────
1. Motion Detection       — downscaled adjacent-frame diff, OpenCL GPU
2. Segment Extraction     — stream-copy per segment (lossless, fast)
3. Concatenation          — timestamp-clean stitch
4. Post-process + Encode  — single FFmpeg pass:
     deflicker → eq (contrast/sat/brightness) → hqdn3d denoise →
     tmix blur  → setpts speedup → transpose (Shorts) → watermark →
     music mix  → libx264/NVENC → -t hard cap

Why one pass? Re-encoding degrades quality. Everything after concat
happens in a single command so there is never a second generation loss.

Why deflicker before speedup? The deflicker filter compares adjacent frames.
After speedup those frames are "virtual" — the comparison is meaningless.

Why tmix before setpts? Motion blur blends N real source frames into one.
Applied after speedup it would blend already-virtual frames with no effect.

MOTION TAB
──────────
Sensitivity 1–3  for night-vision or subtle nestling twitches.
Sensitivity 6–8  when wind/leaves/IR-cut cause false detections.
ROI: draw a rectangle on the preview to ignore motion outside the nest.
MOG2: secondary background model — reduces false positives, ~10% slower.

COLOUR TAB
──────────
Deflicker 5  is ideal for most cameras.  0 = off.
Contrast 1.05–1.15, Saturation 1.05–1.10 give a pleasant look.
Motion Blur -1 = automatic (recommended).  Only applied to 60s Short.

KEYBOARD SHORTCUTS
──────────────────
Ctrl+O  browse files   Ctrl+S  start   Ctrl+C  cancel

YOUTUBE UPLOAD
──────────────
See Advanced tab for step-by-step Google Cloud Console setup.

LOGS
────
log/processor_log.txt      — full rotating debug log (all modules)
log/session_YYYYMMDD_*.txt — per-session human-readable summary

Repository: https://github.com/{GITHUB_REPO}
Version: {VERSION}
""")
        box.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════
    # Settings persist
    # ══════════════════════════════════════════════════════════════════════
    def _collect_settings(self) -> dict:
        music = {k: v for k, v in self._music_paths.items()}
        return {
            # Motion
            "sensitivity":          int(self.sens_var.get()),
            "white_threshold":      int(self.white_var.get()),
            "black_threshold":      int(self.black_var.get()),
            "segment_padding":      round(float(self.pad_var.get()),  3),
            "frame_skip":           int(self.skip_var.get()),
            "merge_gap":            round(float(self.merge_var.get()), 2),
            "use_mog2":             bool(self.mog2_var.get()),
            "mog2_learning_rate":   round(float(self.mog2_var2.get()), 4),
            "roi":                  self._roi,
            # Colour
            "deflicker_size":       int(self.deflicker_var.get()),
            "contrast":             round(float(self.contrast_var.get()), 3),
            "brightness":           round(float(self.brightness_var.get()), 3),
            "saturation":           round(float(self.sat_var.get()), 3),
            "motion_blur_frames":   int(self.blur_var.get()),
            "denoise":              bool(self.denoise_var.get()),
            "quality":              QUALITY_LABELS.index(self.quality_var.get()),
            # Music
            "music_volume":         round(float(self.vol_var.get()), 2),
            "music_paths":          music,
            # Advanced
            "output_dir":           self._output_dir,
            "watermark_text":       self.watermark_entry.get().strip() or None,
            "custom_ffmpeg_args":   self.ffmpeg_entry.get().strip() or None,
            "update_channel":       self.channel_var.get(),
        }

    def _apply_settings(self, s: dict) -> None:
        def _sv(var, key, default):
            var.set(s.get(key, default))

        _sv(self.sens_var,      "sensitivity",       DEFAULTS["sensitivity"])
        _sv(self.white_var,     "white_threshold",   DEFAULTS["white_threshold"])
        _sv(self.black_var,     "black_threshold",   DEFAULTS["black_threshold"])
        _sv(self.pad_var,       "segment_padding",   DEFAULTS["segment_padding"])
        _sv(self.skip_var,      "frame_skip",        DEFAULTS["frame_skip"])
        _sv(self.merge_var,     "merge_gap",         DEFAULTS["merge_gap"])
        _sv(self.mog2_var,      "use_mog2",          DEFAULTS["use_mog2"])
        _sv(self.mog2_var2,     "mog2_learning_rate",DEFAULTS["mog2_learning_rate"])
        _sv(self.deflicker_var, "deflicker_size",    DEFAULTS["deflicker_size"])
        _sv(self.contrast_var,  "contrast",          DEFAULTS["contrast"])
        _sv(self.brightness_var,"brightness",        DEFAULTS["brightness"])
        _sv(self.sat_var,       "saturation",        DEFAULTS["saturation"])
        _sv(self.blur_var,      "motion_blur_frames",DEFAULTS["motion_blur_frames"])
        _sv(self.denoise_var,   "denoise",           DEFAULTS["denoise"])
        _sv(self.vol_var,       "music_volume",      DEFAULTS["music_volume"])

        qi = max(0, min(3, int(s.get("quality", DEFAULTS["quality"]))))
        self.quality_var.set(QUALITY_LABELS[qi])

        self._output_dir = s.get("output_dir")
        self.out_dir_label.configure(text=self._output_dir or "Same as input")
        self.channel_var.set(s.get("update_channel", DEFAULTS["update_channel"]))

        self.watermark_entry.delete(0, "end")
        if s.get("watermark_text"):
            self.watermark_entry.insert(0, s["watermark_text"])
        self.ffmpeg_entry.delete(0, "end")
        if s.get("custom_ffmpeg_args"):
            self.ffmpeg_entry.insert(0, s["custom_ffmpeg_args"])

        self._roi = s.get("roi")
        self._update_roi_status()

        mp = s.get("music_paths", {})
        for k in ("default", "60", "720", "3600"):
            v = mp.get(k) or mp.get(str(k))
            self._music_paths[k] = v
            if k in self._music_labels:
                self._music_labels[k].configure(
                    text=os.path.basename(v) if v else "No file")

    def _save_settings(self) -> None:
        try:
            with open(SETTINGS_FILE, "w") as fh:
                json.dump(self._collect_settings(), fh, indent=2)
            log_session("Settings saved")
            messagebox.showinfo("Saved", "Settings saved.")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not save settings:\n{exc}")

    def _load_settings(self) -> None:
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE) as fh:
                    self._apply_settings(json.load(fh))
                log_session("Settings loaded")
            else:
                self._apply_settings(DEFAULTS)
        except Exception as exc:
            logger.warning(f"Load settings failed: {exc}")
            self._apply_settings(DEFAULTS)

    def _reset_defaults(self) -> None:
        self._apply_settings(DEFAULTS)
        log_session("Settings reset to defaults")

    def _select_output_dir(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._output_dir = d
            self.out_dir_label.configure(text=d)

    # ── Music ──────────────────────────────────────────────────────────────
    def _pick_music(self, key, lbl) -> None:
        p = filedialog.askopenfilename(
            filetypes=[("Audio", "*.mp3 *.wav *.ogg *.aac *.m4a"), ("All", "*.*")]
        )
        if p:
            self._music_paths[key] = p
            lbl.configure(text=os.path.basename(p))

    def _clear_music(self, key, lbl) -> None:
        self._music_paths[key] = None
        lbl.configure(text="No file")

    # ── Presets ────────────────────────────────────────────────────────────
    def _load_presets(self) -> None:
        try:
            if os.path.exists(PRESETS_FILE):
                with open(PRESETS_FILE) as fh:
                    self.presets = json.load(fh)
                self.preset_combo.configure(values=list(self.presets.keys()))
        except Exception as exc:
            logger.warning(f"Load presets failed: {exc}")
            self.presets = {}

    def _save_preset(self) -> None:
        name = self.preset_name.get().strip()
        if not name:
            messagebox.showwarning("Preset", "Enter a name first.")
            return
        self.presets[name] = self._collect_settings()
        with open(PRESETS_FILE, "w") as fh:
            json.dump(self.presets, fh, indent=2)
        self.preset_combo.configure(values=list(self.presets.keys()))
        self.preset_combo.set(name)
        messagebox.showinfo("Preset", f"Saved '{name}'.")

    def _load_preset(self) -> None:
        name = self.preset_combo.get()
        if name in self.presets:
            self._apply_settings(self.presets[name])
        else:
            messagebox.showwarning("Preset", "Select a preset from the list.")

    def _delete_preset(self) -> None:
        name = self.preset_combo.get()
        if name in self.presets:
            del self.presets[name]
            with open(PRESETS_FILE, "w") as fh:
                json.dump(self.presets, fh, indent=2)
            self.preset_combo.configure(values=list(self.presets.keys()))
            self.preset_combo.set("")

    # ══════════════════════════════════════════════════════════════════════
    # ROI Picker
    # ══════════════════════════════════════════════════════════════════════
    def _start_roi_picker(self) -> None:
        """Show a full-size window of the first frame for ROI picking."""
        if not self.input_files:
            messagebox.showwarning("ROI", "Browse a file first.")
            return

        cap = cv2.VideoCapture(self.input_files[0])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("ROI", "Could not read first frame.")
            return

        h, w = frame.shape[:2]
        # Scale to fit screen
        max_dim = 700
        scale   = min(max_dim / w, max_dim / h, 1.0)
        dw, dh  = int(w * scale), int(h * scale)
        frame_r = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))

        # Popup window
        win = ctk.CTkToplevel(self.root)
        win.title("Draw ROI — click and drag, then click Save")
        win.grab_set()

        # Use a plain tk.Canvas for mouse event support
        canvas = tk.Canvas(win, width=dw, height=dh, cursor="crosshair",
                           bg="black", bd=0, highlightthickness=0)
        canvas.pack(padx=10, pady=10)

        tk_img = tk.PhotoImage(
            data=self._pil_to_tk_data(img_pil, dw, dh)
        )
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        state = {"x0": None, "y0": None, "rect": None}

        def _down(e):
            state["x0"], state["y0"] = e.x, e.y
            if state["rect"]:
                canvas.delete(state["rect"])

        def _drag(e):
            if state["x0"] is None: return
            if state["rect"]: canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                state["x0"], state["y0"], e.x, e.y,
                outline="#7c3aed", width=2,
            )

        def _save():
            if state["x0"] is None:
                messagebox.showwarning("ROI", "Draw a rectangle first.", parent=win)
                return
            x0 = min(state["x0"], canvas.winfo_pointerx() - canvas.winfo_rootx())
            y0 = min(state["y0"], canvas.winfo_pointery() - canvas.winfo_rooty())
            x1 = max(state["x0"], canvas.winfo_pointerx() - canvas.winfo_rootx())
            y1 = max(state["y0"], canvas.winfo_pointery() - canvas.winfo_rooty())
            # Convert to normalised fractions
            rx = max(0.0, x0 / dw)
            ry = max(0.0, y0 / dh)
            rw = min(1.0, (x1 - x0) / dw)
            rh = min(1.0, (y1 - y0) / dh)
            if rw < 0.01 or rh < 0.01:
                messagebox.showwarning("ROI", "ROI too small.", parent=win)
                return
            self._roi = {"x": round(rx,4), "y": round(ry,4),
                         "w": round(rw,4), "h": round(rh,4)}
            self._update_roi_status()
            log_session(f"ROI set: {self._roi}")
            logger.info(f"ROI set: {self._roi}")
            win.destroy()

        def _clear():
            self._roi = None
            self._update_roi_status()
            win.destroy()

        canvas.bind("<ButtonPress-1>",   _down)
        canvas.bind("<B1-Motion>",       _drag)

        btns = ctk.CTkFrame(win); btns.pack(pady=8)
        ctk.CTkButton(btns, text="✔  Save ROI",   command=_save,
                      fg_color="#16a34a", font=self.F_BTN).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="✕  Clear ROI",  command=_clear,
                      fg_color="gray40", font=self.F_BTN).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Cancel",         command=win.destroy,
                      font=self.F_BTN).pack(side="left", padx=6)

        win.mainloop()

    @staticmethod
    def _pil_to_tk_data(img: Image.Image, w: int, h: int) -> str:
        """Convert PIL image to Tkinter-compatible PhotoImage data string."""
        img = img.resize((w, h), Image.LANCZOS)
        # Write as PPM into bytes
        import io
        buf = io.BytesIO()
        img.save(buf, format="PPM")
        data = buf.getvalue()
        # Tk PhotoImage can load PPM directly from bytes (via -data is base64)
        import base64
        return base64.b64encode(data).decode()

    def _update_roi_status(self) -> None:
        if hasattr(self, "roi_status"):
            if self._roi:
                r = self._roi
                self.roi_status.configure(
                    text=f"ROI: {r['x']:.2f},{r['y']:.2f} "
                         f"→ {r['w']:.2f}×{r['h']:.2f}",
                    text_color="#7c3aed")
            else:
                self.roi_status.configure(text="ROI: off", text_color="gray60")

    # ══════════════════════════════════════════════════════════════════════
    # File handling + Preview
    # ══════════════════════════════════════════════════════════════════════
    def browse_files(self) -> None:
        files = filedialog.askopenfilenames(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                       ("All", "*.*")]
        )
        if files:
            self._add_files(list(files))

    def _on_drop(self, event) -> None:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
        paths = self.root.tk.splitlist(event.data)
        self._add_files([p for p in paths if Path(p).suffix.lower() in exts])

    def _add_files(self, paths: List[str]) -> None:
        new = [p for p in paths if os.path.exists(p) and p not in self.input_files]
        if not new: return
        first_add = not self.input_files
        self.input_files.extend(new)
        self.file_label.configure(
            text=", ".join(os.path.basename(p) for p in self.input_files))
        self.start_button.configure(state="normal")
        if first_add:
            self._init_preview(self.input_files[0])
        log_session(f"Files added: {[os.path.basename(p) for p in new]}")

    def _init_preview(self, path: str) -> None:
        if self.preview_cap: self._stop_preview()
        self.preview_cap = cv2.VideoCapture(path)
        if self.preview_cap.isOpened():
            total = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._preview_fps = max(1.0, self.preview_cap.get(cv2.CAP_PROP_FPS))
            self.preview_slider.configure(to=max(1, total - 1))
            self.preview_slider.set(0)
            self.preview_button.configure(state="normal")
            self.roi_button.configure(state="normal")
            self._seek_preview(0)
        else:
            self.preview_cap = None
            self.preview_button.configure(state="disabled")
            self.roi_button.configure(state="disabled")

    def _toggle_preview(self) -> None:
        if self.preview_running: self._stop_preview()
        else:                    self._start_preview()

    def _start_preview(self) -> None:
        if not self.preview_cap or not self.preview_cap.isOpened():
            if self.input_files: self._init_preview(self.input_files[0])
            if not self.preview_cap: return
        self.preview_running = True
        self.preview_button.configure(text="■  Stop")
        self.preview_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.preview_thread.start()

    def _stop_preview(self) -> None:
        self.preview_running = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.5)
            self.preview_thread = None
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        self.preview_button.configure(text="▶  Preview", state="disabled")
        self.roi_button.configure(state="disabled")
        self.preview_label.configure(image=self.blank_img, text="")
        with self.preview_queue.mutex:
            self.preview_queue.queue.clear()

    def _read_frames(self) -> None:
        interval = 1.0 / getattr(self, "_preview_fps", 30)
        while self.preview_running and self.preview_cap and self.preview_cap.isOpened():
            t0  = time.time()
            idx = int(self.preview_slider.get())
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = self.preview_cap.read()
            if not ret: continue
            frame = cv2.resize(frame, (310, 196), interpolation=cv2.INTER_AREA)
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ci    = ctk.CTkImage(light_image=img, dark_image=img, size=(310, 196))
            try:
                self.preview_queue.put_nowait(
                    (ci, min(idx + 1, int(self.preview_slider.cget("to"))))
                )
            except queue.Full:
                pass
            time.sleep(max(0, interval - (time.time() - t0)))

    def _update_preview(self) -> None:
        if self.preview_running:
            try:
                img, nxt = self.preview_queue.get_nowait()
                self.preview_label.configure(image=img, text="")
                self.preview_image = img
                self.preview_slider.set(nxt)
            except queue.Empty:
                pass
        self.root.after(33, self._update_preview)

    def _seek_preview(self, val) -> None:
        if not self.preview_running and self.preview_cap and self.preview_cap.isOpened():
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(float(val))))
            ret, frame = self.preview_cap.read()
            if ret:
                frame = cv2.resize(frame, (310, 196), interpolation=cv2.INTER_AREA)
                img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ci    = ctk.CTkImage(light_image=img, dark_image=img, size=(310, 196))
                self.preview_label.configure(image=ci, text="")
                self.preview_image = ci

    # ══════════════════════════════════════════════════════════════════════
    # Processing
    # ══════════════════════════════════════════════════════════════════════
    def start_processing(self) -> None:
        if not self.input_files:
            messagebox.showwarning("No Files", "Browse a video first.")
            return
        selected: List[Tuple[str, int]] = []
        if self.gen_60s.get():  selected.append(("60s",   59))
        if self.gen_12m.get():  selected.append(("12min", 720))
        if self.gen_1h.get():   selected.append(("1h",    3600))
        c = self.custom_duration.get().strip()
        if c:
            try:
                d = int(c)
                if d > 0: selected.append((f"{d}s", d))
            except ValueError:
                messagebox.showerror("Error", "Custom duration must be a positive integer.")
                return
        if not selected:
            messagebox.showwarning("Nothing selected",
                                   "Enable at least one output duration.")
            return

        files_to_process = list(self.input_files)
        self.input_files.clear()
        self.file_label.configure(text="No files selected")
        self.start_button.configure(state="disabled")

        for w in self.progress_frame.winfo_children(): w.destroy()
        for w in self.output_frame.winfo_children():   w.destroy()
        self.progress_rows  = {}
        self.analytics_data = []

        for f in files_to_process:
            row = ctk.CTkFrame(self.progress_frame); row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=os.path.basename(f), width=200,
                         anchor="w", font=self.F_BODY).pack(side="left", padx=4)
            status = ctk.CTkLabel(row, text="Queued", width=180,
                                  anchor="w", font=self.F_PROG)
            status.pack(side="left", padx=4)
            bar = ctk.CTkProgressBar(row, width=210, height=14)
            bar.pack(side="left", padx=4); bar.set(0)
            pct = ctk.CTkLabel(row, text="0%", width=40, anchor="e", font=self.F_PROG)
            pct.pack(side="left", padx=2)
            cb  = ctk.CTkButton(row, text="✕", width=32,
                                command=lambda _f=f: self._cancel_file(_f),
                                fg_color="gray40", font=self.F_PROG)
            cb.pack(side="right", padx=4)
            self.progress_rows[f] = {"status": status, "progress": bar,
                                     "pct": pct, "cancel": cb}

        for w in (self.sw_60s, self.sw_12m, self.sw_1h,
                  self.browse_button, self.start_button):
            w.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.cancel_button.configure(state="normal")
        self.overall_bar.set(0)
        self.overall_label.configure(text="Starting…")

        self._cancel_flag.clear()
        self._pause_flag.clear()
        self._paused    = False
        self._processing= True
        self._proc_start = time.time()

        threading.Thread(
            target=self._processing_thread,
            args=(files_to_process, selected),
            daemon=True,
        ).start()
        log_session(
            f"Processing started: {len(files_to_process)} file(s) "
            f"→ {[s[0] for s in selected]}"
        )

    def _build_config(self) -> dict:
        s = self._collect_settings()
        mp_int: dict = {}
        for k, v in self._music_paths.items():
            try:    mp_int[int(k)] = v
            except (ValueError, TypeError): mp_int[k] = v
        return {
            **s,
            "use_gpu":      True,
            "cpu_threads":  self.cpu_threads,
            "music_paths":  mp_int,
        }

    def _processing_thread(
        self, files: List[str], selected: List[Tuple[str, int]]
    ) -> None:
        config   = self._build_config()
        n_total  = len(files) * len(selected)
        done_tasks = 0

        for f in files:
            if self._cancel_flag.is_set():
                self.queue.put(("canceled", f, "Canceled")); continue
            try:
                self._process_one(f, selected, config, done_tasks, n_total)
            except Exception as exc:
                logger.exception(f"Unexpected error on {f}: {exc}")
                self.queue.put(("canceled", f, str(exc)))
            done_tasks += len(selected)

        self.queue.put(("all_done", time.time() - self._proc_start))

    def _process_one(
        self, f: str, selected: List, config: dict, done_tasks: int, n_total: int,
    ) -> None:
        self.queue.put(("status",   f, "Detecting motion…"))
        self.queue.put(("progress", f, 0, 0))

        def _det_cb(pct: float) -> None:
            while self._pause_flag.is_set() and not self._cancel_flag.is_set():
                time.sleep(0.1)
            self.queue.put(("progress", f, pct * 0.5, pct * 0.5))
            self.queue.put(("status",   f, f"Detecting motion {pct:.0f}%"))
            overall = ((done_tasks + pct / 100 * 0.5) / max(n_total, 1)) * 100
            self.queue.put(("overall",  overall, f"Motion detection {pct:.0f}%"))

        detector = MotionDetector(config)
        detector.set_cancel_flag(self._cancel_flag)
        t_det = time.time()
        segments, stats = detector.detect_motion(f, _det_cb)
        det_elapsed = time.time() - t_det

        if self._cancel_flag.is_set():
            self.queue.put(("canceled", f, "Canceled")); return
        if not segments:
            self.queue.put(("canceled", f, "No motion detected")); return

        log_session(
            f"Detection done: {os.path.basename(f)} | "
            f"segs={stats.get('motion_segments',0)} "
            f"motion={stats.get('motion_duration',0):.1f}s "
            f"({stats.get('motion_duration',0)/max(stats.get('duration',1),1)*100:.1f}%) "
            f"elapsed={det_elapsed:.1f}s"
        )

        self.analytics_data.append({
            "file":             os.path.basename(f),
            "video_duration":   stats.get("duration", 0),
            "motion_duration":  stats.get("motion_duration", 0),
            "motion_segments":  stats.get("motion_segments", 0),
            "processing_time":  0,
            "detection_method": stats.get("detection_method", "—"),
        })

        base      = os.path.splitext(f)[0]
        out_dir   = self._output_dir or os.path.dirname(f) or "."
        proc_start = time.time()

        for i, (task_name, duration) in enumerate(selected):
            if self._cancel_flag.is_set():
                self.queue.put(("canceled", f, "Canceled")); return

            out_file = os.path.join(out_dir, f"{Path(base).name}_{task_name}.mp4")
            os.makedirs(out_dir, exist_ok=True)
            self.queue.put(("status", f, f"Generating {task_name}…"))

            phase_off  = 50 + i * (50 / len(selected))
            phase_size = 50 / len(selected)

            def _vid_cb(pct: float, _o=phase_off, _s=phase_size,
                        _dt=done_tasks, _i=i) -> None:
                while self._pause_flag.is_set() and not self._cancel_flag.is_set():
                    time.sleep(0.1)
                local   = _o + pct * _s / 100
                overall = ((_dt + _i + pct / 100) / max(n_total, 1)) * 100
                # ETA
                elapsed = time.time() - self._proc_start
                eta_txt = ""
                if overall > 1:
                    eta_sec = elapsed / (overall / 100) - elapsed
                    eta_txt = f" | ETA {int(eta_sec//60)}m {int(eta_sec%60)}s"
                self.queue.put(("progress", f, local, local))
                self.queue.put(("overall",  overall,
                                f"Generating {task_name} {pct:.0f}%{eta_txt}"))

            cfg = dict(config)
            music = (self._music_paths.get(str(duration))
                     or self._music_paths.get("default"))

            processor = VideoProcessor(cfg)
            processor.set_cancel_flag(self._cancel_flag)
            self._active_proc = processor

            ok = processor.create_timelapse(
                f, segments, out_file,
                target_length=duration, music_path=music,
                progress_callback=_vid_cb,
            )
            self._active_proc = None

            if self._cancel_flag.is_set():
                self.queue.put(("canceled", f, "Canceled")); return
            if ok:
                log_session(f"Output: {os.path.basename(out_file)}")
                self.queue.put(("task_done", f, task_name, out_file))
            else:
                self.queue.put(("canceled", f, f"FFmpeg error on {task_name}")); return

        if self.analytics_data:
            self.analytics_data[-1]["processing_time"] = time.time() - proc_start
        self.queue.put(("progress", f, 100, 100))

    # ── Cancel / Pause ─────────────────────────────────────────────────────
    def _toggle_pause(self) -> None:
        if self._paused:
            self._paused = False; self._pause_flag.clear()
            self.pause_button.configure(text="⏸  Pause")
        else:
            self._paused = True; self._pause_flag.set()
            self.pause_button.configure(text="▶  Resume")

    def _cancel_processing(self) -> None:
        if not messagebox.askyesno("Cancel", "Cancel all processing?"): return
        self._cancel_flag.set()
        if self._active_proc: self._active_proc.cancel()

    def _cancel_file(self, f: str) -> None:
        self._cancel_flag.set()
        if self._active_proc: self._active_proc.cancel()
        if f in self.progress_rows:
            self.progress_rows[f]["status"].configure(text="Canceled")
            self.progress_rows[f]["cancel"].configure(state="disabled")

    # ── Queue processor ────────────────────────────────────────────────────
    def _process_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                mtype, *args = msg

                if mtype == "status":
                    f, text = args
                    if f in self.progress_rows:
                        self.progress_rows[f]["status"].configure(text=text)

                elif mtype == "progress":
                    f, pct, _ = args
                    if f in self.progress_rows:
                        self.progress_rows[f]["progress"].set(pct / 100)
                        self.progress_rows[f]["pct"].configure(text=f"{pct:.0f}%")

                elif mtype == "overall":
                    pct, label = args
                    self.overall_bar.set(pct / 100)
                    self.overall_label.configure(text=label)

                elif mtype == "task_done":
                    f, task_name, out_file = args
                    self._add_output_row(f, task_name, out_file)
                    if f in self.progress_rows:
                        self.progress_rows[f]["status"].configure(text=f"✅ {task_name} done")
                        self.progress_rows[f]["pct"].configure(text="100%")

                elif mtype == "canceled":
                    f, reason = args
                    if f in self.progress_rows:
                        self.progress_rows[f]["status"].configure(text=f"✕ {reason}")
                        self.progress_rows[f]["progress"].set(0)
                        self.progress_rows[f]["pct"].configure(text="")
                        self.progress_rows[f]["cancel"].configure(state="disabled")

                elif mtype == "upload_progress":
                    f, pct = args
                    if f in self.progress_rows:
                        self.progress_rows[f]["status"].configure(
                            text=f"Uploading {pct:.0f}%")

                elif mtype == "all_done":
                    self._on_all_done(args[0])

        except queue.Empty:
            pass
        self.root.after(50, self._process_queue)

    def _add_output_row(self, input_file: str, task_name: str, out_file: str) -> None:
        row = ctk.CTkFrame(self.output_frame); row.pack(fill="x", pady=2)
        ctk.CTkLabel(
            row, text=f"{os.path.basename(input_file)} → {task_name}",
            anchor="w", font=self.F_BODY,
        ).pack(side="left", padx=6, expand=True, fill="x")
        ctk.CTkButton(row, text="Open", width=66, font=self.F_BTN,
                      command=lambda: self._open_file(out_file)).pack(side="left", padx=2)
        ub = ctk.CTkButton(row, text="Upload to YouTube", width=152, font=self.F_BTN)
        ub.configure(command=lambda b=ub, f=out_file, t=task_name: start_upload(self, f, t, b))
        ub.pack(side="left", padx=2)

    def _on_all_done(self, elapsed: float) -> None:
        self._processing = False
        self._reset_controls()
        self._refresh_analytics()
        m, s = divmod(int(elapsed), 60)
        self.overall_bar.set(1)
        self.overall_label.configure(text=f"✅ All done in {m}m {s}s")
        messagebox.showinfo("Done", f"Processing complete in {m}m {s}s.")
        log_session(f"All done in {elapsed:.1f}s")
        logger.info(f"All done in {elapsed:.1f}s")

    def _reset_controls(self) -> None:
        for w in (self.sw_60s, self.sw_12m, self.sw_1h, self.browse_button):
            w.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        self._pause_flag.clear(); self._paused = False

    # ── Scheduling ─────────────────────────────────────────────────────────
    def _set_schedule(self) -> None:
        t = self.schedule_entry.get().strip()
        try:
            schedule.every().day.at(t).do(self.start_processing)
            messagebox.showinfo("Scheduled", f"Daily processing at {t}.")
            threading.Thread(target=self._run_scheduler, daemon=True).start()
        except Exception as exc:
            messagebox.showerror("Error", f"Invalid HH:MM time: {exc}")

    def _run_scheduler(self) -> None:
        while True: schedule.run_pending(); time.sleep(30)

    # ── Update check ───────────────────────────────────────────────────────
    def _check_updates(self) -> None:
        try:
            ch  = getattr(self, "channel_var", None)
            ch  = ch.get() if ch else "Stable"
            url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{ch}_version.txt"
            r   = requests.get(url, timeout=5); r.raise_for_status()
            latest = r.text.strip()
            if version.parse(latest) > version.parse(VERSION):
                self.root.after(0, lambda: messagebox.showinfo(
                    "Update Available",
                    f"v{latest} available!\nhttps://github.com/{GITHUB_REPO}/releases",
                ))
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _open_file(path: str) -> None:
        try:    os.startfile(path)
        except AttributeError:
            try:    subprocess.call(["open", path])
            except Exception: subprocess.call(["xdg-open", path])

    def _kb_start(self) -> None:
        if self.start_button.cget("state") == "normal": self.start_processing()

    def _kb_cancel(self) -> None:
        if self.cancel_button.cget("state") == "normal": self._cancel_processing()
