"""
Configuration Manager
Handles saving and loading application settings
"""

import json
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration and settings"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.default_settings = {
            'motion_sensitivity': 5,
            'smoothing': 5,
            'min_motion_duration': 0.5,
            'motion_threshold': 25,
            'blur_size': 21,
            'use_gpu': True,
            'cpu_threads': max(1, os.cpu_count() - 1),
            'output_quality': 2,
            'last_input_dir': '',
            'last_output_dir': '',
            'last_music_dir': ''
        }
    
    def load_settings(self) -> Dict:
        """Load settings from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)
                    logger.info("Settings loaded from config file")
                    return settings
            else:
                logger.info("No config file found, using defaults")
                return self.default_settings.copy()
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return self.default_settings.copy()
    
    def save_settings(self, settings: Dict) -> bool:
        """Save settings to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(settings, f, indent=4)
            logger.info("Settings saved to config file")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def get_setting(self, key: str, default=None):
        """Get a specific setting"""
        settings = self.load_settings()
        return settings.get(key, default)
    
    def set_setting(self, key: str, value):
        """Set a specific setting"""
        settings = self.load_settings()
        settings[key] = value
        return self.save_settings(settings)
