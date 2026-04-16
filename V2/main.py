#!/usr/bin/env python3
"""
Bird Box Video Processor  v12.0
Run: python main.py   or double-click Start.bat
"""
import sys

# Windows: enable per-monitor DPI awareness before any Tk window is created
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

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
    root.mainloop()
