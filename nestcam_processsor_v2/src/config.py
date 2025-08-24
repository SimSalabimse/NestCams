"""
Configuration management for NestCam Processor v2.0
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ProcessingSettings(BaseModel):
    """Video processing settings"""

    motion_threshold: int = Field(default=3000, ge=500, le=20000)
    white_threshold: int = Field(default=200, ge=100, le=255)
    black_threshold: int = Field(default=50, ge=0, le=100)
    clip_limit: float = Field(default=1.0, ge=0.2, le=5.0)
    saturation_multiplier: float = Field(default=1.1, ge=0.5, le=2.0)
    output_resolution: str = Field(default="1920x1080")
    batch_size: int = Field(default=4, ge=1, le=16)
    worker_processes: int = Field(default=2, ge=1, le=8)

    @field_validator("output_resolution")
    @classmethod
    def validate_resolution(cls, v):
        if "x" not in v:
            raise ValueError("Resolution must be in format WIDTHxHEIGHT")
        width, height = v.split("x")
        if not (width.isdigit() and height.isdigit()):
            raise ValueError("Width and height must be numbers")
        return v


class AudioSettings(BaseModel):
    """Audio settings"""

    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    music_paths: Dict[str, Optional[str]] = Field(default_factory=dict)


class UploadSettings(BaseModel):
    """YouTube upload settings"""

    client_secrets_path: str = Field(default="client_secrets.json")
    privacy_status: str = Field(
        default="unlisted", pattern="^(public|private|unlisted)$"
    )
    max_retries: int = Field(default=10, ge=1, le=20)
    chunk_size: int = Field(default=512 * 1024, ge=1024)


class AppConfig(BaseModel):
    """Main application configuration"""

    version: str = "2.0.0"
    debug: bool = False

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data"
    )
    log_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "logs"
    )
    output_dir: Optional[Path] = None

    # Processing settings
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)

    # Update settings
    update_channel: str = Field(default="Stable", pattern="^(Stable|Beta)$")

    class Config:
        arbitrary_types_allowed = True

    def save_to_file(self, path: Optional[Path] = None):
        """Save configuration to JSON file"""
        if path is None:
            path = self.data_dir / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from JSON file"""
        if path is None:
            base_dir = Path(__file__).parent.parent.parent
            path = base_dir / "data" / "settings.json"

        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        return cls()


# Global configuration instance
config = AppConfig.load_from_file()
