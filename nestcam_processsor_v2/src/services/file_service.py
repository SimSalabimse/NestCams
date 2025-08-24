"""
File service for handling file operations
"""

from pathlib import Path
import shutil


class FileService:
    def __init__(self, config):
        self.config = config

    def save_music_file(self, uploaded_file):
        """Save uploaded music file"""
        # Simple implementation - you can expand this
        music_dir = Path("music")
        music_dir.mkdir(exist_ok=True)
        file_path = music_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
