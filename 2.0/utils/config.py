import yaml
import os
from typing import Dict, Any


class Config:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.defaults = {
            "motion_sensitivity": 0.5,
            "min_area": 500,
            "smoothing": True,
            "output_dir": "./output",
            "youtube_client_secrets": "client_secrets.json",
            "update_check": False,
            "github_repo": "",
            "processing_mode": "fast",  # "fast" or "quality"
            "output_quality": "high",  # "high", "medium", "low"
            "motion_algorithm": "mog2",  # "mog2", "knn", "simple"
            "min_segment_duration": 0.5,  # seconds
            "frame_subsample": 5,
            "motion_buffer": 2.0,
            "output_format": "mp4",  # "mp4", "webm", "avi"
            "use_gpu": False,
        }
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                try:
                    loaded = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    print(
                        f"Warning: failed to parse '{self.config_file}'. Using defaults.\n{e}"
                    )
                    return self.defaults.copy()
                return {**self.defaults, **loaded}
        return self.defaults.copy()

    def save_config(self):
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save_config()
