import customtkinter as ctk
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except ImportError:
    TkinterDnD = None
    DND_FILES = None
    import logging
    logging.warning("tkinterdnd2 not available, drag-and-drop disabled")

from ui import VideoProcessorApp

if __name__ == "__main__":
    if TkinterDnD:
        root = TkinterDnD.Tk()
    else:
        root = ctk.CTk()
    app = VideoProcessorApp(root)
    root.mainloop()