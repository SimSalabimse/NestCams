"""
Real-Time Folder Monitor — watches a folder and triggers processing for new videos.
"""

import os
import time
import logging
from pathlib import Path
from typing import Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".MP4", ".MOV"}


class _VideoHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable):
        self.callback = callback
        self._seen: set = set()

    def on_created(self, event):
        if event.is_directory:
            return
        p = event.src_path
        if Path(p).suffix not in VIDEO_EXTENSIONS:
            return
        if p in self._seen:
            return
        self._seen.add(p)
        # Wait for file to finish writing
        time.sleep(3)
        if self._file_stable(p):
            logger.info(f"New video detected: {p}")
            self.callback(p)

    @staticmethod
    def _file_stable(path: str, wait: int = 3) -> bool:
        try:
            s1 = os.path.getsize(path)
            time.sleep(wait)
            return os.path.getsize(path) == s1
        except OSError:
            return False


class RealTimeMonitor:
    def __init__(self, watch_folder: str, callback: Callable):
        self.watch_folder = watch_folder
        self.callback = callback
        self._observer = None

    def start(self) -> bool:
        if not os.path.exists(self.watch_folder):
            logger.error(f"Folder not found: {self.watch_folder}")
            return False
        handler = _VideoHandler(self.callback)
        self._observer = Observer()
        self._observer.schedule(handler, self.watch_folder, recursive=False)
        self._observer.start()
        logger.info(f"Monitoring: {self.watch_folder}")
        return True

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        logger.info("Monitoring stopped")

    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()