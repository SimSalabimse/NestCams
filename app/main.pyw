# main.pyw
import sys
import os
try:
    from tkinterdnd2 import TkinterDnD
    root_base = TkinterDnD.Tk
except ImportError:
    import tkinter as tk
    root_base = tk.Tk

import customtkinter as ctk
from ui import VideoProcessorApp

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

if __name__ == "__main__":
    root = root_base()
    app = VideoProcessorApp(root)
    root.mainloop()