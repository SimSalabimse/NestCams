"""
Real-Time Monitoring Module
Monitors folder for new videos and processes automatically
"""

import os
import time
import logging
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class VideoFileHandler(FileSystemEventHandler):
    """Handles new video file events"""
    
    def __init__(self, callback: Callable, video_extensions: list = None):
        self.callback = callback
        self.video_extensions = video_extensions or ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        self.processing = set()
    
    def on_created(self, event):
        """Called when a file is created"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        ext = Path(file_path).suffix.lower()
        
        if ext in self.video_extensions:
            # Wait a bit to ensure file is fully written
            time.sleep(2)
            
            # Check file is not currently being written
            if self._is_file_complete(file_path):
                if file_path not in self.processing:
                    self.processing.add(file_path)
                    logger.info(f"New video detected: {file_path}")
                    self.callback(file_path)
    
    def _is_file_complete(self, file_path: str, wait_time: int = 5) -> bool:
        """Check if file is completely written"""
        try:
            initial_size = os.path.getsize(file_path)
            time.sleep(wait_time)
            final_size = os.path.getsize(file_path)
            return initial_size == final_size
        except:
            return False


class RealTimeMonitor:
    """Real-time folder monitoring for automatic processing"""
    
    def __init__(self, watch_folder: str, process_callback: Callable):
        """
        Initialize monitor
        
        Args:
            watch_folder: Folder to monitor
            process_callback: Function to call with new video path
        """
        self.watch_folder = watch_folder
        self.process_callback = process_callback
        self.observer = None
        self.running = False
        
        logger.info(f"Real-time monitor initialized for: {watch_folder}")
    
    def start(self):
        """Start monitoring"""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        if not os.path.exists(self.watch_folder):
            logger.error(f"Watch folder does not exist: {self.watch_folder}")
            return False
        
        event_handler = VideoFileHandler(self.process_callback)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.watch_folder, recursive=False)
        self.observer.start()
        self.running = True
        
        logger.info(f"Monitoring started: {self.watch_folder}")
        return True
    
    def stop(self):
        """Stop monitoring"""
        if not self.running:
            return
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        self.running = False
        logger.info("Monitoring stopped")
    
    def is_running(self) -> bool:
        """Check if monitor is running"""
        return self.running


class ScheduledProcessor:
    """Process videos on a schedule"""
    
    def __init__(self, watch_folder: str, process_callback: Callable, 
                 interval_hours: int = 24):
        """
        Initialize scheduled processor
        
        Args:
            watch_folder: Folder to check
            process_callback: Function to call with video paths
            interval_hours: Hours between checks
        """
        self.watch_folder = watch_folder
        self.process_callback = process_callback
        self.interval_hours = interval_hours
        self.running = False
        self.last_check = 0
    
    def start(self):
        """Start scheduled processing"""
        self.running = True
        logger.info(f"Scheduled processor started (interval: {self.interval_hours}h)")
        
        while self.running:
            current_time = time.time()
            
            if current_time - self.last_check >= self.interval_hours * 3600:
                self._check_and_process()
                self.last_check = current_time
            
            time.sleep(300)  # Check every 5 minutes if it's time
    
    def stop(self):
        """Stop scheduled processing"""
        self.running = False
        logger.info("Scheduled processor stopped")
    
    def _check_and_process(self):
        """Check folder and process any videos"""
        try:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
            videos = []
            
            for file in os.listdir(self.watch_folder):
                if Path(file).suffix.lower() in video_extensions:
                    full_path = os.path.join(self.watch_folder, file)
                    videos.append(full_path)
            
            if videos:
                logger.info(f"Found {len(videos)} videos to process")
                self.process_callback(videos)
            else:
                logger.info("No videos found in watch folder")
                
        except Exception as e:
            logger.error(f"Error checking folder: {e}")
