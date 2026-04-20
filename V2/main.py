#!/usr/bin/env python3
"""Bird Box Video Processor  v13.0 — entry point."""

import sys
import os

# ── DPI awareness (must happen before any GUI imports) ─────────────────────
if sys.platform == "win32":
    try:
        import ctypes
        # Per-monitor DPI v2 (Windows 10 1703+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
elif sys.platform == "darwin":
    # macOS: Tk is Retina-aware by default; nothing extra needed
    pass
else:
    # Linux / X11: set GDK scaling if not already set
    if "GDK_SCALE" not in os.environ:
        os.environ.setdefault("GDK_DPI_SCALE", "1")

# ── Drag-and-drop support (optional) ───────────────────────────────────────
try:
    from tkinterdnd2 import TkinterDnD
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

import customtkinter as ctk
from ui import VideoProcessorApp


if __name__ == "__main__":
    if _HAS_DND:
        root = TkinterDnD.Tk()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
    else:
        root = ctk.CTk()

    app = VideoProcessorApp(root, has_dnd=_HAS_DND)

    def _on_close():
        from utils import stop_logging
        stop_logging()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
