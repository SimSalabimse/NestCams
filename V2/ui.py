"""
Bird Box Video Processor  v13.0  —  GUI
═══════════════════════════════════════════════════════════════════════════════

Tabs:
  🏠 Home          — drop zone, one-click presets, viral button, fun facts
  🎬 Create        — file browser, durations, preview, ROI, controls
  ✨ Effects       — film grain, vignette, colour grade, LUT, pacing
  📱 Social        — platform export, captions, hashtags, thumbnails
  🔍 Motion        — all detection parameters + MOG2
  🎨 Colour        — deflicker, eq, blur, denoise, quality
  🎵 Music         — per-duration music paths + volume
  📦 Batch         — folder batch processing with resume
  📊 Analytics     — per-file table + session log viewer
  ⚙️ Advanced      — output dir, watermark, presets, templates, scheduling
  ❓ Help          — pipeline order, keyboard shortcuts
"""

import json
import os
import queue
import threading
import time
import subprocess
import webbrowser
import random
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
from PIL import Image
from tkinter import filedialog, messagebox

from motion_detector       import MotionDetector
from video_processor       import VideoProcessor
from batch_processor       import BatchProcessor
from social_media_exporter import SocialMediaExporter
from template_manager      import TemplateManager
from thumbnail_generator   import ThumbnailGenerator
from caption_generator     import CaptionGenerator
from auto_tuner            import AutoTuner
from youtube_upload        import start_upload
from utils                 import log_session, ToolTip, random_bird_fact

try:
    from tkinterdnd2 import DND_FILES
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

import logging
logger = logging.getLogger(__name__)

VERSION      = "13.0.0"
GITHUB_REPO  = "SimSalabimse/NestCams"
SETTINGS_FILE = "settings.json"
PRESETS_FILE  = "presets.json"

QUALITY_LABELS = ["Low (faster)", "Medium", "High", "Maximum (slower)"]
COLOR_GRADES   = ["None", "GoldenHour", "Misty", "Dramatic", "Pastel", "NightGlow", "NaturalVivid"]
CAPTION_STYLES = ["Minimal", "FunFacts", "StoryMode"]
HASHTAG_SETS   = ["BirdLovers", "NatureASMR", "NestCamDaily", "ViralWildlife", "BirdTok"]
PACING_MODES   = ["Cinematic", "FastCut", "StoryDriven"]
PLATFORMS      = ["YouTubeShorts", "TikTok", "InstagramReels", "Facebook", "X"]

DEFAULTS: dict = {
    "sensitivity": 5, "white_threshold": 225, "black_threshold": 35,
    "segment_padding": 0.3, "frame_skip": 2, "merge_gap": 0.8,
    "min_motion_duration": 0.4, "roi": None, "use_mog2": False,
    "mog2_learning_rate": 0.005,
    "deflicker_size": 5, "contrast": 1.0, "brightness": 0.0,
    "saturation": 1.0, "motion_blur_frames": -1, "denoise": True,
    "include_original_audio": False,
    "cinematic_mode": True, "color_grade_preset": "None",
    "film_grain": 0.0, "vignette_strength": 0.0, "lut_path": None,
    "pacing_mode": "Cinematic",
    "quality": 2, "use_gpu": True, "watermark_text": None,
    "custom_ffmpeg_args": None,
    "music_volume": 0.5,
    "music_paths": {"default": None, "60": None, "720": None, "3600": None},
    "platform_targets": [], "auto_caption": True,
    "caption_style": "FunFacts", "hashtag_set": "BirdLovers",
    "thumbnail_style": "ActionFreeze",
    "output_dir": None, "update_channel": "Stable", "language": "en",
}

QUICK_PRESETS = {
    "🎬 Cinematic Short":  "Baby Birds First Flight",
    "😂 Funny Moments":    "Funny Moments",
    "🌧️ Rainy Drama":     "Rainy Day Drama",
    "🌅 Golden Hour":      "Golden Hour Magic",
    "🌙 Night Vision":     "Night Glow",
    "🚀 Viral Short":      "Viral Short",
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
        self.root.geometry("1180x860")
        self.root.resizable(True, True)
        log_session(f"App started v{VERSION}")

        self.F_BODY  = ctk.CTkFont(size=13)
        self.F_HEAD  = ctk.CTkFont(size=14, weight="bold")
        self.F_BTN   = ctk.CTkFont(size=13)
        self.F_PROG  = ctk.CTkFont(size=12)
        self.F_MONO  = ctk.CTkFont(family="Courier New", size=11)
        self.F_LARGE = ctk.CTkFont(size=20, weight="bold")

        self.input_files:    List[str]   = []
        self.progress_rows:  Dict        = {}
        self.analytics_data: List[dict]  = []
        self.presets:        Dict        = {}
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

        self._music_paths: Dict[str, Optional[str]] = {
            k: None for k in ("default", "60", "720", "3600")
        }
        self._output_dir:  Optional[str]  = None
        self._roi:         Optional[dict] = None
        self._lut_path:    Optional[str]  = None
        self._batch_in_dir:  Optional[str] = None
        self._batch_out_dir: Optional[str] = None

        self._template_mgr = TemplateManager()
        self._caption_gen  = CaptionGenerator()
        self._thumb_gen    = ThumbnailGenerator()
        self._exporter     = SocialMediaExporter()

        self.blank_img = ctk.CTkImage(
            light_image=Image.new("RGB", (320, 200), (18, 18, 18)),
            dark_image= Image.new("RGB", (320, 200), (18, 18, 18)),
            size=(320, 200),
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

        self.root.bind("<Control-o>",       lambda _: self.browse_files())
        self.root.bind("<Control-s>",       lambda _: self._kb_start())
        self.root.bind("<Control-c>",       lambda _: self._kb_cancel())
        self.root.bind("<Control-Shift-V>", lambda _: self._apply_viral_preset())

        self.root.after(50,    self._process_queue)
        self.root.after(33,    self._update_preview)
        self.root.after(15000, self._update_fun_fact)
        threading.Thread(target=self._check_updates, daemon=True).start()

    # ── System check ──────────────────────────────────────────────────────
    def _check_system(self) -> None:
        self.opencl_ok = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
        self.nvenc_ok  = False
        try:
            r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                               capture_output=True, text=True, timeout=5)
            self.nvenc_ok = "h264_nvenc" in r.stdout
        except Exception:
            pass
        self.cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        log_session(f"System: opencl={self.opencl_ok} nvenc={self.nvenc_ok} "
                    f"threads={self.cpu_threads} ram={ram_gb:.1f}GB")

    # ── Tab scaffold ──────────────────────────────────────────────────────
    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self.root)
        self.tabs.pack(pady=8, padx=8, fill="both", expand=True)
        for t in ("🏠 Home", "🎬 Create", "✨ Effects", "📱 Social",
                  "🔍 Motion", "🎨 Colour", "🎵 Music",
                  "📦 Batch", "📊 Analytics", "⚙️ Advanced", "❓ Help"):
            self.tabs.add(t)
        self._build_home_tab()
        self._build_create_tab()
        self._build_effects_tab()
        self._build_social_tab()
        self._build_motion_tab()
        self._build_colour_tab()
        self._build_music_tab()
        self._build_batch_tab()
        self._build_analytics_tab()
        self._build_advanced_tab()
        self._build_help_tab()

    # ══════════════════════════════════════════════════════════════════════
    # HOME TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_home_tab(self) -> None:
        tab = self.tabs.tab("🏠 Home")

        hero = ctk.CTkFrame(tab, fg_color="#1a2e1a", corner_radius=12)
        hero.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(hero, text="🐦  Bird Box → Viral Nature Content",
                     font=self.F_LARGE, text_color="#f59e0b").pack(pady=(12, 2))
        ctk.CTkLabel(hero,
                     text="Drop 24-hour recordings here · One click → Cinematic Shorts for TikTok, Reels & Shorts",
                     font=self.F_BODY, text_color="#86efac").pack(pady=(0, 12))

        dz = ctk.CTkFrame(tab, fg_color="#111827", corner_radius=10,
                          border_width=2, border_color="#374151")
        dz.pack(fill="x", padx=10, pady=4)
        hint = " · Drag & Drop videos here" if self.has_dnd else ""
        ctk.CTkLabel(dz, text=f"📂  Browse or drop video files{hint}",
                     font=self.F_HEAD, text_color="#9ca3af").pack(pady=10)
        ctk.CTkButton(dz, text="Browse Files  (Ctrl+O)", command=self.browse_files,
                      font=self.F_BTN, fg_color="#1d4ed8", hover_color="#1e40af",
                      height=40).pack(pady=(0, 12))

        ctk.CTkLabel(tab, text="Quick-Start Presets", font=self.F_HEAD).pack(
            anchor="w", padx=14, pady=(8, 2))
        prow = ctk.CTkFrame(tab); prow.pack(fill="x", padx=10, pady=2)
        colors = ["#7c3aed", "#059669", "#dc2626", "#d97706", "#0284c7", "#db2777"]
        for i, (label, tmpl) in enumerate(QUICK_PRESETS.items()):
            ctk.CTkButton(
                prow, text=label, width=165, height=44,
                fg_color=colors[i % len(colors)],
                hover_color=colors[i % len(colors)],
                font=self.F_BTN,
                command=lambda t=tmpl: self._apply_template_by_name(t),
            ).pack(side="left", padx=4, pady=4)

        ctk.CTkButton(
            tab, text="🚀  Generate Viral Version  (Ctrl+Shift+V)",
            command=self._apply_viral_preset,
            fg_color="#dc2626", hover_color="#b91c1c",
            font=self.F_LARGE, height=52,
        ).pack(fill="x", padx=10, pady=8)

        self._fact_label = ctk.CTkLabel(
            tab, text=f"💡  {random_bird_fact()}",
            font=self.F_PROG, text_color="#6b7280", wraplength=900)
        self._fact_label.pack(anchor="w", padx=14, pady=4)

        ctk.CTkLabel(tab, text="Recent Files", font=self.F_HEAD).pack(
            anchor="w", padx=14, pady=(8, 2))
        self.recent_frame = ctk.CTkScrollableFrame(tab, height=70)
        self.recent_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(self.recent_frame, text="No recent files yet.",
                     font=self.F_PROG, text_color="gray60").pack(pady=8)

    def _update_fun_fact(self) -> None:
        if hasattr(self, "_fact_label"):
            self._fact_label.configure(text=f"💡  {random_bird_fact()}")
        self.root.after(15000, self._update_fun_fact)

    # ══════════════════════════════════════════════════════════════════════
    # CREATE TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_create_tab(self) -> None:
        tab = self.tabs.tab("🎬 Create")

        parts = (["OpenCL motion"] if self.opencl_ok else []) + \
                (["NVENC encode"]   if self.nvenc_ok  else [])
        btxt = ("✅  GPU active — " + ", ".join(parts)) if parts else \
               "ℹ️  CPU mode — update GPU drivers for OpenCL / NVENC"
        ctk.CTkLabel(tab, text=btxt,
                     fg_color="#16a34a" if parts else "#d97706",
                     corner_radius=6, text_color="white", padx=10,
                     font=self.F_BODY).pack(fill="x", padx=10, pady=(6, 4))

        top = ctk.CTkFrame(tab); top.pack(fill="x", padx=10, pady=4)

        # Left column
        left = ctk.CTkFrame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        hint = " · Drag & Drop" if self.has_dnd else ""
        self.file_label = ctk.CTkLabel(left, text=f"No files selected{hint}",
                                       font=self.F_BODY)
        self.file_label.pack(pady=6)

        self.browse_button = ctk.CTkButton(left, text="Browse Files  (Ctrl+O)",
                                           command=self.browse_files, font=self.F_BTN)
        self.browse_button.pack(pady=4)

        dur = ctk.CTkFrame(left); dur.pack(pady=6)
        self.gen_60s = tk.BooleanVar(value=True)
        self.gen_12m = tk.BooleanVar(value=True)
        self.gen_1h  = tk.BooleanVar(value=True)
        self.sw_60s  = ctk.CTkSwitch(dur, text="60s",   variable=self.gen_60s, font=self.F_BODY)
        self.sw_12m  = ctk.CTkSwitch(dur, text="12min", variable=self.gen_12m, font=self.F_BODY)
        self.sw_1h   = ctk.CTkSwitch(dur, text="1hr",   variable=self.gen_1h,  font=self.F_BODY)
        for sw in (self.sw_60s, self.sw_12m, self.sw_1h):
            sw.pack(side="left", padx=10)
        ToolTip(self.sw_60s, "60-second vertical Short (auto-rotated 90°)")
        ToolTip(self.sw_12m, "12-minute highlights reel")
        ToolTip(self.sw_1h,  "1-hour full reel")

        cf = ctk.CTkFrame(left); cf.pack(pady=4)
        ctk.CTkLabel(cf, text="Custom (s):", font=self.F_BODY).pack(side="left", padx=4)
        self.custom_duration = ctk.CTkEntry(cf, width=90,
                                            placeholder_text="e.g. 300", font=self.F_BODY)
        self.custom_duration.pack(side="left")

        bf = ctk.CTkFrame(left); bf.pack(pady=8)
        self.start_button = ctk.CTkButton(
            bf, text="▶  Start Processing", command=self.start_processing,
            state="disabled", fg_color="#16a34a", hover_color="#15803d",
            font=self.F_HEAD, height=44)
        self.start_button.pack(side="left", padx=4)
        ToolTip(self.start_button, "Ctrl+S")

        self.autotune_button = ctk.CTkButton(
            bf, text="🔧 AutoTune", command=self._run_autotune,
            state="disabled", fg_color="#7c3aed", hover_color="#6d28d9",
            font=self.F_BTN, height=44)
        self.autotune_button.pack(side="left", padx=4)
        ToolTip(self.autotune_button,
                "Automatically find the best motion sensitivity for this video.\n"
                "Runs a 3-minute sample analysis.")

        self.pause_button = ctk.CTkButton(
            bf, text="⏸  Pause", command=self._toggle_pause,
            state="disabled", font=self.F_BTN, height=44)
        self.pause_button.pack(side="left", padx=4)

        self.cancel_button = ctk.CTkButton(
            bf, text="✕  Cancel", command=self._cancel_processing,
            state="disabled", fg_color="#dc2626", hover_color="#b91c1c",
            font=self.F_BTN, height=44)
        self.cancel_button.pack(side="left", padx=4)
        ToolTip(self.cancel_button, "Cancel immediately (Ctrl+C)")

        pg = ctk.CTkFrame(left); pg.pack(fill="x", pady=(6, 0), padx=4)
        self.overall_bar = ctk.CTkProgressBar(pg, height=18)
        self.overall_bar.pack(fill="x", padx=4, pady=(4, 2))
        self.overall_bar.set(0)
        self.overall_label = ctk.CTkLabel(pg, text="", font=self.F_PROG,
                                          text_color="gray70")
        self.overall_label.pack(anchor="w", padx=6)

        # Right column — preview
        right = ctk.CTkFrame(top); right.pack(side="right", padx=(6, 0))
        self.preview_label = ctk.CTkLabel(right, image=self.blank_img, text="")
        self.preview_label.pack(padx=6, pady=4)

        self.preview_slider = ctk.CTkSlider(right, from_=0, to=1,
                                            command=self._seek_preview)
        self.preview_slider.pack(fill="x", padx=6)
        self.preview_slider.set(0)

        prow2 = ctk.CTkFrame(right); prow2.pack(pady=4)
        self.preview_button = ctk.CTkButton(
            prow2, text="▶  Preview", command=self._toggle_preview,
            width=120, state="disabled", font=self.F_BTN)
        self.preview_button.pack(side="left", padx=2)
        self.roi_button = ctk.CTkButton(
            prow2, text="📐 ROI", command=self._start_roi_picker,
            width=90, state="disabled", font=self.F_BTN,
            fg_color="#7c3aed", hover_color="#6d28d9")
        self.roi_button.pack(side="left", padx=2)
        ToolTip(self.roi_button,
                "Draw a rectangle on the preview.\nMotion outside it will be ignored.")
        self.roi_status = ctk.CTkLabel(right, text="ROI: off",
                                       font=self.F_PROG, text_color="gray60")
        self.roi_status.pack()

        ctk.CTkLabel(tab, text="Per-file progress:",
                     font=self.F_BODY).pack(anchor="w", padx=14, pady=(6, 0))
        self.progress_frame = ctk.CTkScrollableFrame(tab, height=160)
        self.progress_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(tab, text="Output files:",
                     font=self.F_BODY).pack(anchor="w", padx=14, pady=(4, 0))
        self.output_frame = ctk.CTkScrollableFrame(tab, height=100)
        self.output_frame.pack(fill="x", padx=10, pady=2)

    # ══════════════════════════════════════════════════════════════════════
    # EFFECTS TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_effects_tab(self) -> None:
        tab = self.tabs.tab("✨ Effects")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(f, text="Colour Grade Preset", font=self.F_HEAD).pack(pady=(4, 0))
        self.grade_var = tk.StringVar(value=DEFAULTS["color_grade_preset"])
        ctk.CTkOptionMenu(f, variable=self.grade_var, values=COLOR_GRADES,
                          font=self.F_BTN).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "GoldenHour: warm sunset feel — best for daytime footage\n"
                "Dramatic: high contrast — best for YouTube thumbnails\n"
                "Misty: soft desaturated — best for cloudy days\n"
                "NightGlow: cool IR look — best for night footage")

        ctk.CTkButton(f, text="📁  Load Custom LUT (.cube / .3dl)",
                      command=self._pick_lut, font=self.F_BTN).pack(pady=4)
        self.lut_label = ctk.CTkLabel(f, text="No LUT loaded",
                                      font=self.F_PROG, text_color="gray60")
        self.lut_label.pack()

        sep = ctk.CTkFrame(f, height=2, fg_color="gray30")
        sep.pack(fill="x", padx=20, pady=8)

        def _row(label, lo, hi, steps, default, fmt, tip=""):
            ctk.CTkLabel(f, text=label, font=self.F_BODY).pack(
                pady=(6, 0), anchor="w", padx=10)
            var = tk.DoubleVar(value=default)
            sl  = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps, variable=var)
            sl.pack(pady=2, fill="x", padx=30)
            lbl = ctk.CTkLabel(f, text=fmt.format(default), font=self.F_PROG)
            lbl.pack()
            var.trace_add("write", lambda *_: lbl.configure(text=fmt.format(var.get())))
            if tip: ToolTip(sl, tip)
            return sl, var

        self.grain_sl, self.grain_var = _row(
            "Film Grain  (0 = off · 0.08 = cinematic · 0.2 = vintage)",
            0.0, 0.4, 40, DEFAULTS["film_grain"], "{:.2f}",
            "Adds subtle texture that increases perceived sharpness.\n"
            "Viewers associate grain with professional cinema 🎬")

        self.vignette_sl, self.vignette_var = _row(
            "Vignette  (0 = off · 0.2 = subtle · 0.5 = strong)",
            0.0, 0.6, 60, DEFAULTS["vignette_strength"], "{:.2f}",
            "Darkens edges to draw the eye toward the bird in the centre.\n"
            "Increases watch time by reducing visual distraction.")

        sep2 = ctk.CTkFrame(f, height=2, fg_color="gray30")
        sep2.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(f, text="Pacing Mode", font=self.F_HEAD).pack(pady=(4, 0))
        self.pacing_var = tk.StringVar(value=DEFAULTS["pacing_mode"])
        ctk.CTkOptionMenu(f, variable=self.pacing_var, values=PACING_MODES,
                          font=self.F_BTN).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "FastCut: maximum activity → best for TikTok/Shorts\n"
                "Cinematic: smooth pacing → best for YouTube\n"
                "StoryDriven: keeps longer quiet moments for context")

        self.cinematic_var = tk.BooleanVar(value=DEFAULTS["cinematic_mode"])
        ctk.CTkSwitch(f, text="Cinematic Mode (auto film grain + vignette)",
                      variable=self.cinematic_var, font=self.F_BODY).pack(pady=6)

        sep3 = ctk.CTkFrame(f, height=2, fg_color="gray30")
        sep3.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(f, text="Apply Look Template", font=self.F_HEAD).pack(pady=(4, 0))
        self.effects_tmpl_var = tk.StringVar(value="")
        ctk.CTkOptionMenu(
            f, variable=self.effects_tmpl_var,
            values=self._template_mgr.list_templates(), font=self.F_BTN,
        ).pack(pady=4)
        ctk.CTkButton(f, text="Apply Template",
                      command=lambda: self._apply_template_by_name(
                          self.effects_tmpl_var.get()),
                      font=self.F_BTN).pack(pady=4)

        ctk.CTkButton(f, text="💾  Save Effects Settings",
                      command=self._save_settings, font=self.F_BTN).pack(pady=12)

    # ══════════════════════════════════════════════════════════════════════
    # SOCIAL TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_social_tab(self) -> None:
        tab = self.tabs.tab("📱 Social")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(f, text="Target Platforms", font=self.F_HEAD).pack(pady=(4, 2))
        self._platform_vars: Dict[str, tk.BooleanVar] = {}
        pf = ctk.CTkFrame(f); pf.pack(fill="x", padx=10, pady=4)
        for p in PLATFORMS:
            var = tk.BooleanVar(value=False)
            self._platform_vars[p] = var
            ctk.CTkCheckBox(pf, text=p, variable=var, font=self.F_BODY).pack(
                side="left", padx=10, pady=4)
        ToolTip(pf, "Selected platforms will be re-encoded after each output is complete.\n"
                    "Each platform gets the correct aspect ratio and max duration.")

        sep = ctk.CTkFrame(f, height=2, fg_color="gray30")
        sep.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(f, text="Caption & Hashtags", font=self.F_HEAD).pack(pady=(4, 0))

        self.auto_caption_var = tk.BooleanVar(value=DEFAULTS["auto_caption"])
        ctk.CTkSwitch(f, text="Auto-generate caption + hashtags after processing",
                      variable=self.auto_caption_var, font=self.F_BODY).pack(pady=4)

        ctk.CTkLabel(f, text="Caption Style", font=self.F_BODY).pack(pady=(6, 0))
        self.caption_style_var = tk.StringVar(value=DEFAULTS["caption_style"])
        ctk.CTkOptionMenu(f, variable=self.caption_style_var, values=CAPTION_STYLES,
                          font=self.F_BTN).pack(pady=4)

        ctk.CTkLabel(f, text="Hashtag Set", font=self.F_BODY).pack(pady=(6, 0))
        self.hashtag_set_var = tk.StringVar(value=DEFAULTS["hashtag_set"])
        ctk.CTkOptionMenu(f, variable=self.hashtag_set_var, values=HASHTAG_SETS,
                          font=self.F_BTN).pack(pady=4)

        sep2 = ctk.CTkFrame(f, height=2, fg_color="gray30")
        sep2.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(f, text="Caption Preview", font=self.F_HEAD).pack(pady=(4, 0))
        self.caption_preview = ctk.CTkTextbox(f, height=120, font=self.F_MONO)
        self.caption_preview.pack(fill="x", padx=10, pady=4)
        self.caption_preview.configure(state="disabled")

        br = ctk.CTkFrame(f); br.pack(pady=4)
        ctk.CTkButton(br, text="🔄  Generate Preview",
                      command=self._preview_caption, font=self.F_BTN).pack(side="left", padx=4)
        ctk.CTkButton(br, text="📋  Copy to Clipboard",
                      command=self._copy_caption, font=self.F_BTN).pack(side="left", padx=4)

        sep3 = ctk.CTkFrame(f, height=2, fg_color="gray30")
        sep3.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(f, text="Thumbnail Style", font=self.F_HEAD).pack(pady=(4, 0))
        self.thumb_style_var = tk.StringVar(value=DEFAULTS["thumbnail_style"])
        ctk.CTkOptionMenu(f, variable=self.thumb_style_var,
                          values=["ActionFreeze", "SplitBeforeAfter", "TextOverlay"],
                          font=self.F_BTN).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "ActionFreeze: best frame from highest-activity moment\n"
                "SplitBeforeAfter: side-by-side comparison\n"
                "TextOverlay: best frame with bold text banner")

    def _preview_caption(self) -> None:
        exporter = SocialMediaExporter()
        hashtags = exporter.generate_hashtags(set_name=self.hashtag_set_var.get())
        cap = self._caption_gen.generate(
            stats={"motion_duration": 42.0, "duration": 86400.0,
                   "motion_segments": 18, "date": "today"},
            hashtag_set=self.hashtag_set_var.get(),
            caption_style=self.caption_style_var.get(),
            hashtags=hashtags,
        )
        self.caption_preview.configure(state="normal")
        self.caption_preview.delete("1.0", "end")
        self.caption_preview.insert("end", cap)
        self.caption_preview.configure(state="disabled")

    def _copy_caption(self) -> None:
        try:
            text = self.caption_preview.get("1.0", "end").strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Caption copied to clipboard!")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # MOTION TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_motion_tab(self) -> None:
        tab = self.tabs.tab("🔍 Motion")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        def _row(label, lo, hi, steps, default, fmt, tip=""):
            ctk.CTkLabel(f, text=label, font=self.F_BODY).pack(
                pady=(8, 0), anchor="w", padx=10)
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
            "1–3: catches nestling twitches and tiny wing flicks\n"
            "6–8: use when wind/leaves/IR-cut cause false detections")
        self.skip_sl, self.skip_var = _row(
            "Frame Skip  (analyse every Nth frame)", 1, 8, 7, DEFAULTS["frame_skip"], "{:.0f}",
            "2 is ideal for 30 fps cameras. Raise to speed up detection on very long files.")
        self.pad_sl, self.pad_var = _row(
            "Segment Padding  (seconds of context before/after each event)",
            0.0, 5.0, 50, DEFAULTS["segment_padding"], "{:.2f}s")
        self.merge_sl, self.merge_var = _row(
            "Merge Gap  (gaps shorter than this are merged)",
            0.1, 5.0, 49, DEFAULTS["merge_gap"], "{:.1f}s")
        self.white_sl, self.white_var = _row(
            "White Frame Threshold  (skip overexposed frames)",
            100, 255, 155, DEFAULTS["white_threshold"], "{:.0f}")
        self.black_sl, self.black_var = _row(
            "Black Frame Threshold  (skip underexposed/night frames)",
            0, 80, 80, DEFAULTS["black_threshold"], "{:.0f}")

        sep = ctk.CTkFrame(f, height=2, fg_color="gray30"); sep.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(f, text="Background Subtraction (MOG2)",
                     font=self.F_HEAD).pack(pady=(4, 0))
        self.mog2_var = tk.BooleanVar(value=DEFAULTS["use_mog2"])
        ctk.CTkSwitch(f, text="Enable MOG2 secondary check",
                      variable=self.mog2_var, font=self.F_BODY).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "Builds a background model. Both the diff-test AND MOG2 must agree.\n"
                "Reduces false positives at the cost of ~10% extra processing time.")

        self.mog2_sl, self.mog2_var2 = _row(
            "MOG2 Learning Rate  (how quickly background adapts)",
            0.001, 0.05, 49, DEFAULTS["mog2_learning_rate"], "{:.4f}",
            "Lower = slower adaptation (better for static cameras). 0.005 is ideal.")

        ctk.CTkButton(f, text="💾  Save Motion Settings",
                      command=self._save_settings, font=self.F_BTN).pack(pady=12)

    # ══════════════════════════════════════════════════════════════════════
    # COLOUR TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_colour_tab(self) -> None:
        tab = self.tabs.tab("🎨 Colour")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        def _row(label, lo, hi, steps, default, fmt, tip=""):
            ctk.CTkLabel(f, text=label, font=self.F_BODY).pack(
                pady=(8, 0), anchor="w", padx=10)
            var = tk.DoubleVar(value=default)
            sl  = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps, variable=var)
            sl.pack(pady=2, fill="x", padx=30)
            lbl = ctk.CTkLabel(f, text=fmt.format(default), font=self.F_PROG)
            lbl.pack()
            var.trace_add("write", lambda *_: lbl.configure(text=fmt.format(var.get())))
            if tip: ToolTip(sl, tip)
            return sl, var

        self.deflicker_sl, self.deflicker_var = _row(
            "Deflicker Strength  (sliding window — 0 = off · 5 = ideal)",
            0, 15, 15, DEFAULTS["deflicker_size"], "{:.0f}")
        self.contrast_sl, self.contrast_var = _row(
            "Contrast  (1.0 = neutral · 1.10 = punchy)",
            0.8, 1.5, 70, DEFAULTS["contrast"], "{:.2f}")
        self.brightness_sl, self.brightness_var = _row(
            "Brightness  (0 = neutral · positive = brighter)",
            -0.3, 0.3, 60, DEFAULTS["brightness"], "{:.2f}")
        self.sat_sl, self.sat_var = _row(
            "Saturation  (1.0 = neutral · 1.10 = warmer colours)",
            0.5, 1.8, 130, DEFAULTS["saturation"], "{:.2f}")

        sep = ctk.CTkFrame(f, height=2, fg_color="gray30"); sep.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(f, text="Quality & Smoothing", font=self.F_HEAD).pack(pady=(4, 0))

        ctk.CTkLabel(f, text="Output Quality", font=self.F_BODY).pack(pady=(6, 0))
        self.quality_var = tk.StringVar(value=QUALITY_LABELS[DEFAULTS["quality"]])
        ctk.CTkOptionMenu(f, variable=self.quality_var, values=QUALITY_LABELS,
                          font=self.F_BTN).pack(pady=4)

        self.denoise_var = tk.BooleanVar(value=DEFAULTS["denoise"])
        ctk.CTkSwitch(f, text="Apply hqdn3d light denoising",
                      variable=self.denoise_var, font=self.F_BODY).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "Removes sensor noise and codec artefacts before encoding.\n"
                "Improves perceived sharpness with negligible blurring.")

        self.blur_sl, self.blur_var = _row(
            "Motion Blur Frames  (-1 = auto · 0 = off · 3–8 = manual)",
            -1, 12, 13, DEFAULTS["motion_blur_frames"], "{:.0f}",
            "-1: automatically chosen from speed factor (recommended).\n"
            "Only applied to the 60-second Short output.")

        self.orig_audio_var = tk.BooleanVar(value=DEFAULTS["include_original_audio"])
        ctk.CTkSwitch(f, text="Include original nest audio at low volume (authenticity mix)",
                      variable=self.orig_audio_var, font=self.F_BODY).pack(pady=4)
        ToolTip(f.winfo_children()[-1],
                "Mixes nest sounds quietly under the music track.\n"
                "Adds authenticity — popular for ASMR-style nature content.")

        ctk.CTkButton(f, text="💾  Save Colour Settings",
                      command=self._save_settings, font=self.F_BTN).pack(pady=12)

    # ══════════════════════════════════════════════════════════════════════
    # MUSIC TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_music_tab(self) -> None:
        tab = self.tabs.tab("🎵 Music")
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
            ctk.CTkButton(r, text="Browse…", width=90, font=self.F_BTN,
                          command=lambda k=key, l=lbl: self._pick_music(k, l)
                          ).pack(side="right", padx=2)
            ctk.CTkButton(r, text="Clear", width=60, font=self.F_BTN,
                          command=lambda k=key, l=lbl: self._clear_music(k, l)
                          ).pack(side="right", padx=2)
            self._music_labels[key] = lbl

        ctk.CTkLabel(
            f,
            text="Music is looped to fill the target duration, then faded out over the last 2 seconds.",
            text_color="gray", font=self.F_PROG,
        ).pack(pady=10)

    # ══════════════════════════════════════════════════════════════════════
    # BATCH TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_batch_tab(self) -> None:
        tab = self.tabs.tab("📦 Batch")
        f   = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(f, text="Batch Processing", font=self.F_HEAD).pack(pady=(4, 8))

        ctk.CTkLabel(f, text="Input folder:", font=self.F_BODY).pack(anchor="w", padx=10)
        bf = ctk.CTkFrame(f); bf.pack(fill="x", padx=10, pady=2)
        self.batch_in_label = ctk.CTkLabel(bf, text="Not set", font=self.F_BODY)
        self.batch_in_label.pack(side="left", padx=4, expand=True, fill="x")
        ctk.CTkButton(bf, text="Browse…", width=90, font=self.F_BTN,
                      command=self._select_batch_input).pack(side="right", padx=4)

        ctk.CTkLabel(f, text="Output folder:", font=self.F_BODY).pack(
            anchor="w", padx=10, pady=(8, 0))
        bof = ctk.CTkFrame(f); bof.pack(fill="x", padx=10, pady=2)
        self.batch_out_label = ctk.CTkLabel(bof, text="Same as input", font=self.F_BODY)
        self.batch_out_label.pack(side="left", padx=4, expand=True, fill="x")
        ctk.CTkButton(bof, text="Browse…", width=90, font=self.F_BTN,
                      command=self._select_batch_output).pack(side="right", padx=4)

        ctk.CTkLabel(f, text="Durations:", font=self.F_BODY).pack(anchor="w", padx=10, pady=(8, 0))
        bdr = ctk.CTkFrame(f); bdr.pack(fill="x", padx=10, pady=2)
        self.batch_60s = tk.BooleanVar(value=True)
        self.batch_12m = tk.BooleanVar(value=True)
        self.batch_1h  = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(bdr, text="60s",   variable=self.batch_60s, font=self.F_BODY).pack(side="left", padx=10)
        ctk.CTkCheckBox(bdr, text="12min", variable=self.batch_12m, font=self.F_BODY).pack(side="left", padx=10)
        ctk.CTkCheckBox(bdr, text="1hr",   variable=self.batch_1h,  font=self.F_BODY).pack(side="left", padx=10)

        self.batch_resume_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(f, text="Resume — skip already-completed files",
                      variable=self.batch_resume_var, font=self.F_BODY).pack(pady=8)
        ToolTip(f.winfo_children()[-1],
                "Files with a .done.json sidecar are skipped.\n"
                "Safe to re-run after a crash — no double-processing.")

        ctk.CTkButton(
            f, text="▶  Start Batch Processing", command=self._start_batch,
            fg_color="#16a34a", hover_color="#15803d",
            font=self.F_HEAD, height=44,
        ).pack(pady=8)

        self.batch_bar = ctk.CTkProgressBar(f, height=14)
        self.batch_bar.pack(fill="x", padx=10, pady=2)
        self.batch_bar.set(0)
        self.batch_label = ctk.CTkLabel(f, text="", font=self.F_PROG,
                                        text_color="gray70")
        self.batch_label.pack()

    def _select_batch_input(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._batch_in_dir = d
            self.batch_in_label.configure(text=d)

    def _select_batch_output(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._batch_out_dir = d
            self.batch_out_label.configure(text=d)

    def _start_batch(self) -> None:
        if not self._batch_in_dir:
            messagebox.showwarning("Batch", "Select an input folder first."); return
        durations = []
        if self.batch_60s.get(): durations.append(59)
        if self.batch_12m.get(): durations.append(720)
        if self.batch_1h.get():  durations.append(3600)
        if not durations:
            messagebox.showwarning("Batch", "Select at least one duration."); return

        out_dir = self._batch_out_dir or self._batch_in_dir
        cfg     = self._build_config()

        def _run():
            bp = BatchProcessor(cfg)
            bp.set_cancel_flag(self._cancel_flag)

            def _pcb(idx, total, pct):
                self.queue.put(("batch_progress",
                                (idx * 100 + pct) / max(total, 1),
                                f"File {idx+1}/{total} — {pct:.0f}%"))

            result = bp.process_folder(
                self._batch_in_dir, out_dir,
                durations   = durations,
                resume_from = self.batch_resume_var.get() or None,
                progress_cb = _pcb,
                status_cb   = lambda m: self.queue.put(("batch_progress", None, m)),
            )
            self.queue.put(("batch_done", result))

        self._cancel_flag.clear()
        threading.Thread(target=_run, daemon=True).start()
        log_session(f"Batch started: {self._batch_in_dir}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTICS TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_analytics_tab(self) -> None:
        tab = self.tabs.tab("📊 Analytics")
        ctk.CTkLabel(tab, text="Session Analytics",
                     font=self.F_HEAD).pack(pady=(10, 4))
        self.analytics_scroll = ctk.CTkScrollableFrame(tab, height=240)
        self.analytics_scroll.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(tab, text="Session Log  (last 200 lines):",
                     font=self.F_BODY).pack(anchor="w", padx=14, pady=(8, 0))
        self.log_viewer = ctk.CTkTextbox(tab, font=self.F_MONO, height=180)
        self.log_viewer.pack(fill="both", expand=True, padx=10, pady=4)
        ctk.CTkButton(tab, text="🔄 Refresh Log",
                      command=self._refresh_log_viewer, font=self.F_BTN).pack(pady=4)

    def _refresh_analytics(self) -> None:
        for w in self.analytics_scroll.winfo_children():
            w.destroy()
        if not self.analytics_data:
            ctk.CTkLabel(self.analytics_scroll,
                         text="No data yet — process a video first.",
                         text_color="gray", font=self.F_PROG).pack(pady=10)
            return
        hdr = ctk.CTkFrame(self.analytics_scroll); hdr.pack(fill="x", pady=1)
        for col, w in [("File", 200), ("Video", 70), ("Motion", 90),
                       ("%", 55), ("Segs", 55), ("Proc", 70), ("Method", 120)]:
            ctk.CTkLabel(hdr, text=col, width=w, anchor="w",
                         font=self.F_BODY, text_color="gray70").pack(side="left", padx=3)
        for d in self.analytics_data:
            dur = d.get("video_duration", 0)
            mot = d.get("motion_duration", 0)
            pct = mot / max(dur, 1) * 100
            r   = ctk.CTkFrame(self.analytics_scroll); r.pack(fill="x", pady=1)
            for txt, ww in [
                (d.get("file", "?")[:26], 200), (f"{dur:.0f}s", 70),
                (f"{mot:.1f}s", 90), (f"{pct:.1f}%", 55),
                (str(d.get("motion_segments", 0)), 55),
                (f"{d.get('processing_time', 0):.1f}s", 70),
                (d.get("detection_method", "—"), 120),
            ]:
                ctk.CTkLabel(r, text=txt, width=ww, anchor="w",
                             font=self.F_PROG).pack(side="left", padx=3)

    def _refresh_log_viewer(self) -> None:
        import glob as _glob
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
        files   = sorted(_glob.glob(os.path.join(log_dir, "session_*.txt")))
        if not files: return
        with open(files[-1], encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        self.log_viewer.configure(state="normal")
        self.log_viewer.delete("1.0", "end")
        self.log_viewer.insert("end", "".join(lines[-200:]))
        self.log_viewer.configure(state="disabled")
        self.log_viewer.see("end")

    # ══════════════════════════════════════════════════════════════════════
    # ADVANCED TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_advanced_tab(self) -> None:
        tab = self.tabs.tab("⚙️ Advanced")
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
        self.watermark_entry = ctk.CTkEntry(f, placeholder_text="© My Channel 2024",
                                            font=self.F_BODY)
        self.watermark_entry.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(f, text="Extra FFmpeg arguments (appended to final encode):",
                     font=self.F_BODY).pack(pady=(8, 0), anchor="w")
        self.ffmpeg_entry = ctk.CTkEntry(f, placeholder_text="-vf hue=s=0",
                                         font=self.F_BODY)
        self.ffmpeg_entry.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(f, text="Caption language:", font=self.F_BODY).pack(
            pady=(8, 0), anchor="w")
        self.lang_var = tk.StringVar(value="en")
        ctk.CTkOptionMenu(f, variable=self.lang_var, values=["en", "de", "es"],
                          font=self.F_BTN).pack(pady=2)

        chan = ctk.CTkFrame(f); chan.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(chan, text="Update channel:", font=self.F_BODY).pack(
            side="left", padx=4)
        self.channel_var = tk.StringVar(value=DEFAULTS["update_channel"])
        ctk.CTkOptionMenu(chan, variable=self.channel_var,
                          values=["Stable", "Beta"], font=self.F_BTN).pack(side="left")

        yt = ctk.CTkFrame(f); yt.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(yt, text="YouTube Upload Setup", font=self.F_HEAD).pack(pady=4)
        ctk.CTkLabel(
            yt,
            text=("1. https://console.cloud.google.com\n"
                  "2. Create project → enable 'YouTube Data API v3'\n"
                  "3. Credentials → OAuth 2.0 (Desktop) → Download JSON\n"
                  "4. Rename → client_secrets.json → place in program folder\n"
                  "5. First upload opens browser for one-time authorisation"),
            justify="left", font=self.F_PROG, text_color="gray80",
        ).pack(padx=10, pady=4, anchor="w")
        ctk.CTkButton(yt, text="Open Google Cloud Console", font=self.F_BTN,
                      command=lambda: webbrowser.open(
                          "https://console.cloud.google.com")).pack(pady=4)

        sf = ctk.CTkFrame(f); sf.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(sf, text="Schedule daily processing at (HH:MM):",
                     font=self.F_BODY).pack(pady=4)
        self.schedule_entry = ctk.CTkEntry(sf, placeholder_text="23:30",
                                           width=120, font=self.F_BODY)
        self.schedule_entry.pack(pady=2)
        ctk.CTkButton(sf, text="Set Schedule", command=self._set_schedule,
                      font=self.F_BTN).pack(pady=4)

        pf = ctk.CTkFrame(f); pf.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(pf, text="Settings Presets", font=self.F_HEAD).pack(pady=4)
        self.preset_combo = ctk.CTkComboBox(pf, values=[], width=280, font=self.F_BTN)
        self.preset_combo.pack(pady=4)
        r1 = ctk.CTkFrame(pf); r1.pack(pady=2)
        ctk.CTkButton(r1, text="Load",   command=self._load_preset,
                      font=self.F_BTN).pack(side="left", padx=4)
        ctk.CTkButton(r1, text="Delete", command=self._delete_preset,
                      fg_color="gray40", font=self.F_BTN).pack(side="left", padx=4)
        r2 = ctk.CTkFrame(pf); r2.pack(pady=4)
        self.preset_name = ctk.CTkEntry(r2, placeholder_text="Preset name…",
                                        width=180, font=self.F_BODY)
        self.preset_name.pack(side="left", padx=4)
        ctk.CTkButton(r2, text="Save as Preset",
                      command=self._save_preset, font=self.F_BTN).pack(side="left", padx=4)

        tf = ctk.CTkFrame(f); tf.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(tf, text="Look Template Import / Export",
                     font=self.F_HEAD).pack(pady=4)
        r3 = ctk.CTkFrame(tf); r3.pack(pady=2)
        ctk.CTkButton(r3, text="Import Template JSON",
                      command=self._import_template, font=self.F_BTN).pack(
            side="left", padx=4)
        ctk.CTkButton(r3, text="Export Current as Template",
                      command=self._export_template, font=self.F_BTN).pack(
            side="left", padx=4)

        btns = ctk.CTkFrame(f); btns.pack(pady=12)
        ctk.CTkButton(btns, text="💾  Save All Settings",
                      command=self._save_settings, font=self.F_BTN).pack(
            side="left", padx=6)
        ctk.CTkButton(btns, text="↺  Reset to Defaults",
                      command=self._reset_defaults, fg_color="gray40",
                      font=self.F_BTN).pack(side="left", padx=6)

    # ══════════════════════════════════════════════════════════════════════
    # HELP TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_help_tab(self) -> None:
        tab = self.tabs.tab("❓ Help")
        box = ctk.CTkTextbox(tab, wrap="word", font=self.F_BODY)
        box.pack(fill="both", expand=True, padx=10, pady=10)
        box.insert("end", f"""\
Bird Box Video Processor  v{VERSION}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PIPELINE ORDER  (why it is optimal)
────────────────────────────────────
1. Motion Detection       — GPU-accelerated adjacent-frame diff (640px downscale)
2. Segment Extraction     — stream-copy (lossless, fast, no re-encode)
3. Concatenation          — timestamp-clean stitch
4. Single FFmpeg pass:
     setpts-reset → deflicker → eq (colour grade) → hqdn3d denoise →
     tmix motion blur (BEFORE speedup) → setpts speedup →
     film grain → vignette → LUT/grade →
     transpose (Shorts only) → watermark →
     music mix → libx264/NVENC → -t hard cap

Why one pass?  Re-encoding degrades quality (generation loss).
Why tmix before setpts?  tmix blends real source frames. After speedup those
frames are virtual — the blur has no effect.
Why deflicker before speedup?  deflicker compares adjacent frames.
After speedup they represent multiple originals — the comparison is meaningless.

ENTERTAINMENT FEATURES
──────────────────────
Effects tab:   Colour grade presets, custom LUT, film grain, vignette, pacing modes
Social tab:    Platform export (TikTok, Reels, Shorts, X, Facebook)
               Auto caption + hashtag generation (EN/DE/ES)
               3-variant thumbnail auto-generation
Batch tab:     Whole-folder processing with crash-safe resume
AutoTune:      3-minute sample analysis → optimal sensitivity suggestion

KEYBOARD SHORTCUTS
──────────────────
Ctrl+O            Browse files
Ctrl+S            Start processing
Ctrl+C            Cancel
Ctrl+Shift+V      Apply Viral preset

LOGS
────
log/processor_log.txt       — full rotating debug log (all modules)
log/session_YYYYMMDD*.txt   — per-session human-readable summary

Repository: https://github.com/{GITHUB_REPO}
""")
        box.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════
    # Settings collect / apply / save / load
    # ══════════════════════════════════════════════════════════════════════
    def _collect_settings(self) -> dict:
        return {
            "sensitivity":            int(self.sens_var.get()),
            "white_threshold":        int(self.white_var.get()),
            "black_threshold":        int(self.black_var.get()),
            "segment_padding":        round(float(self.pad_var.get()), 3),
            "frame_skip":             int(self.skip_var.get()),
            "merge_gap":              round(float(self.merge_var.get()), 2),
            "use_mog2":               bool(self.mog2_var.get()),
            "mog2_learning_rate":     round(float(self.mog2_var2.get()), 4),
            "roi":                    self._roi,
            "deflicker_size":         int(self.deflicker_var.get()),
            "contrast":               round(float(self.contrast_var.get()), 3),
            "brightness":             round(float(self.brightness_var.get()), 3),
            "saturation":             round(float(self.sat_var.get()), 3),
            "motion_blur_frames":     int(self.blur_var.get()),
            "denoise":                bool(self.denoise_var.get()),
            "include_original_audio": bool(self.orig_audio_var.get()),
            "quality":                QUALITY_LABELS.index(self.quality_var.get()),
            "color_grade_preset":     self.grade_var.get(),
            "film_grain":             round(float(self.grain_var.get()), 3),
            "vignette_strength":      round(float(self.vignette_var.get()), 3),
            "lut_path":               self._lut_path,
            "pacing_mode":            self.pacing_var.get(),
            "cinematic_mode":         bool(self.cinematic_var.get()),
            "music_volume":           round(float(self.vol_var.get()), 2),
            "music_paths":            dict(self._music_paths),
            "platform_targets":       [p for p, v in self._platform_vars.items() if v.get()],
            "auto_caption":           bool(self.auto_caption_var.get()),
            "caption_style":          self.caption_style_var.get(),
            "hashtag_set":            self.hashtag_set_var.get(),
            "thumbnail_style":        self.thumb_style_var.get(),
            "output_dir":             self._output_dir,
            "watermark_text":         self.watermark_entry.get().strip() or None,
            "custom_ffmpeg_args":     self.ffmpeg_entry.get().strip() or None,
            "update_channel":         self.channel_var.get(),
            "language":               self.lang_var.get(),
        }

    def _apply_settings(self, s: dict) -> None:
        def _sv(var, key, default): var.set(s.get(key, default))
        _sv(self.sens_var,        "sensitivity",            DEFAULTS["sensitivity"])
        _sv(self.white_var,       "white_threshold",        DEFAULTS["white_threshold"])
        _sv(self.black_var,       "black_threshold",        DEFAULTS["black_threshold"])
        _sv(self.pad_var,         "segment_padding",        DEFAULTS["segment_padding"])
        _sv(self.skip_var,        "frame_skip",             DEFAULTS["frame_skip"])
        _sv(self.merge_var,       "merge_gap",              DEFAULTS["merge_gap"])
        _sv(self.mog2_var,        "use_mog2",               DEFAULTS["use_mog2"])
        _sv(self.mog2_var2,       "mog2_learning_rate",     DEFAULTS["mog2_learning_rate"])
        _sv(self.deflicker_var,   "deflicker_size",         DEFAULTS["deflicker_size"])
        _sv(self.contrast_var,    "contrast",               DEFAULTS["contrast"])
        _sv(self.brightness_var,  "brightness",             DEFAULTS["brightness"])
        _sv(self.sat_var,         "saturation",             DEFAULTS["saturation"])
        _sv(self.blur_var,        "motion_blur_frames",     DEFAULTS["motion_blur_frames"])
        _sv(self.denoise_var,     "denoise",                DEFAULTS["denoise"])
        _sv(self.orig_audio_var,  "include_original_audio", DEFAULTS["include_original_audio"])
        _sv(self.vol_var,         "music_volume",           DEFAULTS["music_volume"])
        _sv(self.grain_var,       "film_grain",             DEFAULTS["film_grain"])
        _sv(self.vignette_var,    "vignette_strength",      DEFAULTS["vignette_strength"])
        _sv(self.cinematic_var,   "cinematic_mode",         DEFAULTS["cinematic_mode"])
        _sv(self.auto_caption_var,"auto_caption",           DEFAULTS["auto_caption"])

        qi = max(0, min(3, int(s.get("quality", DEFAULTS["quality"]))))
        self.quality_var.set(QUALITY_LABELS[qi])
        self.grade_var.set(s.get("color_grade_preset", DEFAULTS["color_grade_preset"]))
        self.pacing_var.set(s.get("pacing_mode",        DEFAULTS["pacing_mode"]))
        self.caption_style_var.set(s.get("caption_style", DEFAULTS["caption_style"]))
        self.hashtag_set_var.set(s.get("hashtag_set",    DEFAULTS["hashtag_set"]))
        self.thumb_style_var.set(s.get("thumbnail_style", DEFAULTS["thumbnail_style"]))
        if hasattr(self, "lang_var"):
            self.lang_var.set(s.get("language", "en"))
        if hasattr(self, "channel_var"):
            self.channel_var.set(s.get("update_channel", DEFAULTS["update_channel"]))

        self._output_dir = s.get("output_dir")
        self.out_dir_label.configure(text=self._output_dir or "Same as input")

        self.watermark_entry.delete(0, "end")
        if s.get("watermark_text"):
            self.watermark_entry.insert(0, s["watermark_text"])
        self.ffmpeg_entry.delete(0, "end")
        if s.get("custom_ffmpeg_args"):
            self.ffmpeg_entry.insert(0, s["custom_ffmpeg_args"])

        self._roi = s.get("roi")
        self._update_roi_status()

        self._lut_path = s.get("lut_path")
        if hasattr(self, "lut_label"):
            self.lut_label.configure(
                text=os.path.basename(self._lut_path) if self._lut_path else "No LUT loaded")

        mp = s.get("music_paths", {})
        for k in ("default", "60", "720", "3600"):
            v = mp.get(k) or mp.get(str(k))
            self._music_paths[k] = v
            if k in self._music_labels:
                self._music_labels[k].configure(
                    text=os.path.basename(v) if v else "No file")

        for p, var in self._platform_vars.items():
            var.set(p in s.get("platform_targets", []))

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

    def _select_output_dir(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._output_dir = d
            self.out_dir_label.configure(text=d)

    # ── Music / LUT ───────────────────────────────────────────────────────
    def _pick_music(self, key, lbl) -> None:
        p = filedialog.askopenfilename(
            filetypes=[("Audio", "*.mp3 *.wav *.ogg *.aac *.m4a"), ("All", "*.*")])
        if p:
            self._music_paths[key] = p
            lbl.configure(text=os.path.basename(p))

    def _clear_music(self, key, lbl) -> None:
        self._music_paths[key] = None
        lbl.configure(text="No file")

    def _pick_lut(self) -> None:
        p = filedialog.askopenfilename(
            filetypes=[("LUT", "*.cube *.3dl *.lut"), ("All", "*.*")])
        if p:
            self._lut_path = p
            self.lut_label.configure(text=os.path.basename(p))

    # ── Templates ─────────────────────────────────────────────────────────
    def _apply_template_by_name(self, name: str) -> None:
        tmpl = self._template_mgr.get(name)
        if not tmpl:
            messagebox.showwarning("Template", f"Template not found: {name}"); return
        self._apply_settings({**self._collect_settings(), **tmpl})
        log_session(f"Template applied: {name}")

    def _apply_viral_preset(self) -> None:
        self._apply_template_by_name("Viral Short")
        self.gen_60s.set(True)
        for p, var in self._platform_vars.items():
            var.set(p in ("YouTubeShorts", "TikTok"))
        messagebox.showinfo(
            "Viral Mode 🚀",
            "Viral Short preset applied!\n"
            "60s Short + TikTok + YouTube Shorts enabled.\n"
            "Drop a video and press Start!")

    def _import_template(self) -> None:
        p = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if p:
            name = self._template_mgr.import_template(p)
            if name: messagebox.showinfo("Template", f"Imported: {name}")
            else:    messagebox.showerror("Template", "Import failed.")

    def _export_template(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            with open(p, "w") as fh:
                json.dump({"name": Path(p).stem,
                           "template": self._collect_settings()}, fh, indent=2)
            messagebox.showinfo("Template", f"Exported to {p}")

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
            messagebox.showwarning("Preset", "Enter a name first."); return
        self.presets[name] = self._collect_settings()
        with open(PRESETS_FILE, "w") as fh:
            json.dump(self.presets, fh, indent=2)
        self.preset_combo.configure(values=list(self.presets.keys()))
        self.preset_combo.set(name)
        messagebox.showinfo("Preset", f"Saved '{name}'.")

    def _load_preset(self) -> None:
        name = self.preset_combo.get()
        if name in self.presets: self._apply_settings(self.presets[name])
        else: messagebox.showwarning("Preset", "Select a preset from the list.")

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
        if not self.input_files:
            messagebox.showwarning("ROI", "Browse a file first."); return
        cap = cv2.VideoCapture(self.input_files[0])
        ok, frame = cap.read(); cap.release()
        if not ok:
            messagebox.showerror("ROI", "Could not read first frame."); return

        h, w   = frame.shape[:2]
        scale  = min(700 / w, 700 / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)
        frame_r = cv2.resize(frame, (dw, dh))
        img_pil = Image.fromarray(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))

        win = ctk.CTkToplevel(self.root)
        win.title("Draw ROI — drag then click Save")
        win.grab_set()

        canvas = tk.Canvas(win, width=dw, height=dh, cursor="crosshair",
                           bg="black", bd=0, highlightthickness=0)
        canvas.pack(padx=10, pady=10)

        import io, base64
        buf = io.BytesIO(); img_pil.save(buf, format="PPM")
        tk_img = tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode())
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        state = {"x0": None, "y0": None, "rect": None}

        def _down(e):
            state["x0"], state["y0"] = e.x, e.y
            if state["rect"]: canvas.delete(state["rect"])

        def _drag(e):
            if state["x0"] is None: return
            if state["rect"]: canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                state["x0"], state["y0"], e.x, e.y,
                outline="#7c3aed", width=2)

        def _save():
            if state["x0"] is None:
                messagebox.showwarning("ROI", "Draw a rectangle first.", parent=win); return
            x0 = min(state["x0"], canvas.winfo_pointerx() - canvas.winfo_rootx())
            y0 = min(state["y0"], canvas.winfo_pointery() - canvas.winfo_rooty())
            x1 = max(state["x0"], canvas.winfo_pointerx() - canvas.winfo_rootx())
            y1 = max(state["y0"], canvas.winfo_pointery() - canvas.winfo_rooty())
            rw, rh = (x1 - x0) / dw, (y1 - y0) / dh
            if rw < 0.01 or rh < 0.01:
                messagebox.showwarning("ROI", "ROI too small.", parent=win); return
            self._roi = {"x": round(x0/dw, 4), "y": round(y0/dh, 4),
                         "w": round(rw, 4),     "h": round(rh, 4)}
            self._update_roi_status()
            log_session(f"ROI set: {self._roi}")
            win.destroy()

        canvas.bind("<ButtonPress-1>", _down)
        canvas.bind("<B1-Motion>",     _drag)

        btns = ctk.CTkFrame(win); btns.pack(pady=8)
        ctk.CTkButton(btns, text="✔ Save ROI", command=_save,
                      fg_color="#16a34a", font=self.F_BTN).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="✕ Clear ROI",
                      command=lambda: (
                          setattr(self, "_roi", None),
                          self._update_roi_status(), win.destroy()),
                      fg_color="gray40", font=self.F_BTN).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Cancel",
                      command=win.destroy, font=self.F_BTN).pack(side="left", padx=6)
        win.mainloop()

    def _update_roi_status(self) -> None:
        if hasattr(self, "roi_status"):
            if self._roi:
                r = self._roi
                self.roi_status.configure(
                    text=f"ROI: {r['x']:.2f},{r['y']:.2f} → {r['w']:.2f}×{r['h']:.2f}",
                    text_color="#7c3aed")
            else:
                self.roi_status.configure(text="ROI: off", text_color="gray60")

    # ══════════════════════════════════════════════════════════════════════
    # File handling + Preview
    # ══════════════════════════════════════════════════════════════════════
    def browse_files(self) -> None:
        files = filedialog.askopenfilenames(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                       ("All", "*.*")])
        if files: self._add_files(list(files))

    def _on_drop(self, event) -> None:
        exts  = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
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
        self.autotune_button.configure(state="normal")
        if first_add: self._init_preview(self.input_files[0])
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
            frame = cv2.resize(frame, (320, 200))
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ci    = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 200))
            try:
                self.preview_queue.put_nowait(
                    (ci, min(idx + 1, int(self.preview_slider.cget("to")))))
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
                frame = cv2.resize(frame, (320, 200))
                img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ci    = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 200))
                self.preview_label.configure(image=ci, text="")
                self.preview_image = ci

    # ══════════════════════════════════════════════════════════════════════
    # AutoTune
    # ══════════════════════════════════════════════════════════════════════
    def _run_autotune(self) -> None:
        if not self.input_files:
            messagebox.showwarning("AutoTune", "Browse a file first."); return
        self.autotune_button.configure(state="disabled", text="🔧 Tuning…")

        def _go():
            tuner  = AutoTuner(self._build_config())
            result = tuner.suggest_optimal_settings(
                self.input_files[0],
                progress_callback=lambda p: self.queue.put(
                    ("overall", p, f"AutoTune {p:.0f}%")),
            )
            self.queue.put(("autotune_done", result))

        threading.Thread(target=_go, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════
    # Processing
    # ══════════════════════════════════════════════════════════════════════
    def start_processing(self) -> None:
        if not self.input_files:
            messagebox.showwarning("No Files", "Browse a video first."); return
        selected: List[Tuple[str, int]] = []
        if self.gen_60s.get(): selected.append(("60s",   59))
        if self.gen_12m.get(): selected.append(("12min", 720))
        if self.gen_1h.get():  selected.append(("1h",    3600))
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
                                   "Enable at least one output duration."); return

        files_to_process = list(self.input_files)
        self.input_files.clear()
        self.file_label.configure(text="No files selected")
        self.start_button.configure(state="disabled")
        self.autotune_button.configure(state="disabled")

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
            bar = ctk.CTkProgressBar(row, width=200, height=14)
            bar.pack(side="left", padx=4); bar.set(0)
            pct = ctk.CTkLabel(row, text="0%", width=40, anchor="e", font=self.F_PROG)
            pct.pack(side="left", padx=2)
            self.progress_rows[f] = {"status": status, "progress": bar, "pct": pct}

        for w in (self.sw_60s, self.sw_12m, self.sw_1h, self.browse_button):
            w.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.cancel_button.configure(state="normal")
        self.overall_bar.set(0)
        self.overall_label.configure(text="Starting…")

        self._cancel_flag.clear()
        self._pause_flag.clear()
        self._paused     = False
        self._processing = True
        self._proc_start = time.time()

        threading.Thread(
            target=self._processing_thread,
            args=(files_to_process, selected),
            daemon=True,
        ).start()
        log_session(f"Processing started: {len(files_to_process)} file(s) "
                    f"→ {[s[0] for s in selected]}")

    def _build_config(self) -> dict:
        s  = self._collect_settings()
        mp: dict = {}
        for k, v in self._music_paths.items():
            try:    mp[int(k)] = v
            except (ValueError, TypeError): mp[k] = v
        return {**s, "use_gpu": True, "cpu_threads": self.cpu_threads, "music_paths": mp}

    def _processing_thread(self, files, selected) -> None:
        config     = self._build_config()
        n_total    = len(files) * len(selected)
        done_tasks = 0
        for f in files:
            if self._cancel_flag.is_set():
                self.queue.put(("canceled", f, "Canceled")); continue
            try:
                self._process_one(f, selected, config, done_tasks, n_total)
            except Exception as exc:
                logger.exception(f"Error on {f}: {exc}")
                self.queue.put(("canceled", f, str(exc)))
            done_tasks += len(selected)
        self.queue.put(("all_done", time.time() - self._proc_start))

    def _process_one(self, f, selected, config, done_tasks, n_total) -> None:
        self.queue.put(("status", f, "Detecting motion…"))

        def _det_cb(pct):
            while self._pause_flag.is_set() and not self._cancel_flag.is_set():
                time.sleep(0.1)
            self.queue.put(("progress", f, pct * 0.5, pct * 0.5))
            overall = ((done_tasks + pct / 100 * 0.5) / max(n_total, 1)) * 100
            self.queue.put(("overall", overall, f"Motion detection {pct:.0f}%"))

        detector = MotionDetector(config)
        detector.set_cancel_flag(self._cancel_flag)
        t_det    = time.time()
        segments, stats = detector.detect_motion(f, _det_cb)
        det_elapsed = time.time() - t_det

        if self._cancel_flag.is_set():
            self.queue.put(("canceled", f, "Canceled")); return
        if not segments:
            self.queue.put(("canceled", f, "No motion detected")); return

        log_session(f"Detection done: {os.path.basename(f)} | "
                    f"segs={stats.get('motion_segments',0)} "
                    f"motion={stats.get('motion_duration',0):.1f}s "
                    f"elapsed={det_elapsed:.1f}s")

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

            def _vid_cb(pct, _o=phase_off, _s=phase_size, _dt=done_tasks, _i=i):
                while self._pause_flag.is_set() and not self._cancel_flag.is_set():
                    time.sleep(0.1)
                local   = _o + pct * _s / 100
                overall = ((_dt + _i + pct / 100) / max(n_total, 1)) * 100
                elapsed = time.time() - self._proc_start
                eta_txt = ""
                if overall > 1:
                    eta_sec = elapsed / (overall / 100) - elapsed
                    eta_txt = f" | ETA {int(eta_sec//60)}m {int(eta_sec%60)}s"
                self.queue.put(("progress", f, local, local))
                self.queue.put(("overall",  overall,
                                f"Generating {task_name} {pct:.0f}%{eta_txt}"))

            music = (self._music_paths.get(str(duration)) or
                     self._music_paths.get("default"))
            processor = VideoProcessor(dict(config))
            processor.set_cancel_flag(self._cancel_flag)
            self._active_proc = processor

            ok = processor.create_timelapse(
                f, segments, out_file, target_length=duration,
                music_path=music, progress_callback=_vid_cb,
            )
            self._active_proc = None

            if self._cancel_flag.is_set():
                self.queue.put(("canceled", f, "Canceled")); return

            if ok:
                log_session(f"Output: {os.path.basename(out_file)}")
                self.queue.put(("task_done", f, task_name, out_file, segments, stats))
            else:
                self.queue.put(("canceled", f, f"FFmpeg error on {task_name}")); return

        if self.analytics_data:
            self.analytics_data[-1]["processing_time"] = time.time() - proc_start
        self.queue.put(("progress", f, 100, 100))

    def _post_process_output(self, out_file, task_name, segments, stats) -> None:
        from metadata_handler import MetadataHandler
        meta = MetadataHandler()
        meta.generate_best_thumbnail(out_file)
        meta.embed_metadata(out_file, {**stats, "task": task_name})

        if self.auto_caption_var.get():
            exporter = SocialMediaExporter()
            hashtags = exporter.generate_hashtags(set_name=self.hashtag_set_var.get())
            cap      = self._caption_gen.generate(
                stats={**stats, "date": time.strftime("%Y-%m-%d")},
                hashtag_set   = self.hashtag_set_var.get(),
                caption_style = self.caption_style_var.get(),
                hashtags      = hashtags,
            )
            caption_path = out_file.replace(".mp4", "_caption.txt")
            with open(caption_path, "w", encoding="utf-8") as fh:
                fh.write(cap)

        target_platforms = [p for p, v in self._platform_vars.items() if v.get()]
        if target_platforms:
            out_dir = os.path.dirname(out_file)
            for platform in target_platforms:
                self._exporter.export_for_platform(out_file, platform, {}, out_dir)

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
                    f, task_name, out_file, segments, stats = args
                    self._add_output_row(f, task_name, out_file)
                    if f in self.progress_rows:
                        self.progress_rows[f]["status"].configure(
                            text=f"✅ {task_name} done")
                        self.progress_rows[f]["pct"].configure(text="100%")
                    threading.Thread(
                        target=self._post_process_output,
                        args=(out_file, task_name, segments, stats),
                        daemon=True,
                    ).start()

                elif mtype == "canceled":
                    f, reason = args
                    if f in self.progress_rows:
                        self.progress_rows[f]["status"].configure(text=f"✕ {reason}")
                        self.progress_rows[f]["progress"].set(0)
                        self.progress_rows[f]["pct"].configure(text="")

                elif mtype == "all_done":
                    self._on_all_done(args[0])

                elif mtype == "autotune_done":
                    result = args[0]
                    self._apply_settings({**self._collect_settings(), **result})
                    self.autotune_button.configure(state="normal", text="🔧 AutoTune")
                    messagebox.showinfo(
                        "AutoTune Complete",
                        f"Optimal sensitivity: {result.get('sensitivity','?')}\n"
                        f"Frame skip: {result.get('frame_skip','?')}\n"
                        f"Deflicker: {result.get('deflicker_size','?')}\n"
                        "Settings applied automatically.")

                elif mtype == "batch_progress":
                    pct, label = args
                    if pct is not None:
                        self.batch_bar.set(pct / 100)
                    self.batch_label.configure(text=label)

                elif mtype == "batch_done":
                    result = args[0]
                    self.batch_bar.set(1)
                    self.batch_label.configure(
                        text=f"✅ Done: {result['success']} ok · "
                             f"{result['failed']} failed · {result['skipped']} skipped")
                    messagebox.showinfo(
                        "Batch Complete",
                        f"Processed {result['total']} file(s)\n"
                        f"Success:  {result['success']}\n"
                        f"Failed:   {result['failed']}\n"
                        f"Skipped:  {result['skipped']}")

        except queue.Empty:
            pass
        self.root.after(50, self._process_queue)

    def _add_output_row(self, input_file, task_name, out_file) -> None:
        row = ctk.CTkFrame(self.output_frame); row.pack(fill="x", pady=2)
        ctk.CTkLabel(
            row, text=f"{os.path.basename(input_file)} → {task_name}",
            anchor="w", font=self.F_BODY,
        ).pack(side="left", padx=6, expand=True, fill="x")
        ctk.CTkButton(row, text="Open", width=60, font=self.F_BTN,
                      command=lambda: self._open_file(out_file)).pack(side="left", padx=2)
        ub = ctk.CTkButton(row, text="YouTube", width=90, font=self.F_BTN)
        ub.configure(
            command=lambda b=ub, f=out_file, t=task_name: start_upload(self, f, t, b))
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

    def _reset_controls(self) -> None:
        for w in (self.sw_60s, self.sw_12m, self.sw_1h, self.browse_button):
            w.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        self._pause_flag.clear(); self._paused = False

    # ── Scheduling ──────────────────────────────────────────────────────────
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

    # ── Updates ─────────────────────────────────────────────────────────────
    def _check_updates(self) -> None:
        try:
            ch  = getattr(self, "channel_var", None)
            ch  = ch.get() if ch else "Stable"
            url = (f"https://raw.githubusercontent.com/{GITHUB_REPO}"
                   f"/main/{ch}_version.txt")
            r   = requests.get(url, timeout=5); r.raise_for_status()
            latest = r.text.strip()
            if version.parse(latest) > version.parse(VERSION):
                self.root.after(0, lambda: messagebox.showinfo(
                    "Update Available",
                    f"v{latest} available!\n"
                    f"https://github.com/{GITHUB_REPO}/releases"))
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _open_file(path: str) -> None:
        try:    os.startfile(path)
        except AttributeError:
            try:    subprocess.call(["open", path])
            except Exception: subprocess.call(["xdg-open", path])

    def _kb_start(self) -> None:
        if self.start_button.cget("state") == "normal":
            self.start_processing()

    def _kb_cancel(self) -> None:
        if self.cancel_button.cget("state") == "normal":
            self._cancel_processing()
