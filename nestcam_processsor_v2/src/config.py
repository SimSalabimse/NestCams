"""
Configuration management for NestCam Processor v2.0
"""

from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import json
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor
import tempfile
import shutil
import time

# Load environment variables
load_dotenv()

# GPU Support (CUDA for 4070ti)
try:
    import cupy as cp

    HAS_GPU = True
    print("✅ GPU acceleration enabled (CUDA)")
except ImportError:
    HAS_GPU = False
    print("⚠️ GPU acceleration not available (install cupy for CUDA support)")


# Memory monitoring
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def log_memory_usage(message):
    """Log current memory usage"""
    mem_gb = get_memory_usage()
    # Assuming logger is defined elsewhere or will be added
    # logger.info(f"{message} - Memory: {mem_gb:.2f}GB")
    print(f"{message} - Memory: {mem_gb:.2f}GB")


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
    use_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    chunk_size: int = Field(
        default=500, ge=50, le=2000, description="Frames per processing chunk"
    )
    memory_limit_gb: float = Field(
        default=8.0, ge=1.0, le=64.0, description="Memory limit in GB"
    )

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

    def process_video_streaming(
        self,
        input_path: str,
        output_format: str = "mp4",
        output_dir: Optional[str] = None,
        watermark_text: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        chunk_size: int = 1000,  # Process in chunks of 1000 frames
        use_gpu: bool = True,
    ) -> "ProcessingResult":  # Assuming ProcessingResult is defined elsewhere
        """
        Memory-efficient video processing with GPU acceleration

        Args:
            chunk_size: Number of frames to process at once (reduces memory usage)
            use_gpu: Whether to use GPU acceleration if available
        """
        log_memory_usage("Starting streaming video processing")
        start_time = time.time()

        # Enable GPU if requested and available
        if use_gpu and HAS_GPU:
            self._enable_gpu_acceleration()
            # logger.info("🎯 GPU acceleration enabled for processing") # Assuming logger is defined
            print("🎯 GPU acceleration enabled for processing")

        try:
            # Process video in chunks to minimize memory usage
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Process in chunks
                chunk_outputs = []
                for chunk_start in range(0, total_frames, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_frames)

                    log_memory_usage(f"Processing chunk {chunk_start}-{chunk_end}")

                    # Process chunk and get output file
                    chunk_output = self._process_video_chunk(
                        cap,
                        chunk_start,
                        chunk_end,
                        temp_path,
                        watermark_text,
                        progress_callback,
                    )
                    chunk_outputs.append(chunk_output)

                    # Force garbage collection after each chunk
                    gc.collect()
                    log_memory_usage(f"After chunk {chunk_start}-{chunk_end}")

                cap.release()

                # Combine chunks into final video
                final_output = self._combine_video_chunks(
                    chunk_outputs, output_dir, fps, width, height, output_format
                )

                processing_time = time.time() - start_time
                log_memory_usage("Processing completed")

                return "ProcessingResult"  # Placeholder for ProcessingResult

        except Exception as e:
            # logger.error(f"Streaming processing failed: {e}") # Assuming logger is defined
            print(f"Streaming processing failed: {e}")
            raise

    def _enable_gpu_acceleration(self):
        """Enable GPU acceleration for OpenCV operations"""
        if HAS_GPU:
            # Enable CUDA for various OpenCV operations
            cv2.cuda.setDevice(0)  # Use first GPU
            # logger.info("🎯 GPU acceleration enabled") # Assuming logger is defined
            print("🎯 GPU acceleration enabled")

    def _process_video_chunk(
        self, cap, start_frame, end_frame, temp_dir, watermark_text, progress_callback
    ):
        """Process a chunk of frames with memory optimization"""
        frames = []

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame immediately and clear memory
            processed_frame = self._process_single_frame(frame, watermark_text)

            # Save to temporary file instead of keeping in memory
            frame_path = temp_dir / f"chunk_{start_frame}_{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), processed_frame)

            # Clear frame from memory
            del frame, processed_frame
            if frame_idx % 100 == 0:
                gc.collect()  # Periodic cleanup

        return temp_dir

    def _process_single_frame(self, frame, watermark_text):
        """Process a single frame with GPU acceleration if available"""
        if HAS_GPU and self.processing.use_gpu:
            # GPU-accelerated processing
            return self._process_frame_gpu(frame, watermark_text)
        else:
            # CPU processing
            enhanced = self.enhancer.enhance_frame(
                frame
            )  # Assuming enhancer is defined
            if watermark_text:
                enhanced = self._add_watermark(
                    enhanced, watermark_text
                )  # Assuming _add_watermark is defined
            return enhanced

    def _process_frame_gpu(self, frame, watermark_text):
        """GPU-accelerated frame processing"""
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # GPU-based enhancement (simplified example)
        # In practice, you'd implement GPU versions of your enhancement algorithms
        enhanced_gpu = cv2.cuda.bilateralFilter(gpu_frame, 9, 75, 75)

        # Download from GPU
        enhanced = enhanced_gpu.download()

        if watermark_text:
            enhanced = self._add_watermark(
                enhanced, watermark_text
            )  # Assuming _add_watermark is defined

        return enhanced

    def _add_watermark(self, frame, text):
        """Add a watermark to the frame"""
        # Assuming watermark_font and watermark_color are defined
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, frame.shape[0] - 20)
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 2
        cv2.putText(frame, text, org, font, font_scale, color, thickness)
        return frame

    def _combine_video_chunks(
        self, chunk_outputs, output_dir, fps, width, height, output_format
    ):
        """Combine processed chunks into a final video"""
        # Assuming temp_dir is defined
        final_output_path = (
            Path(output_dir) / f"processed_video.{output_format}"
            if output_dir
            else Path(".") / f"processed_video.{output_format}"
        )
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Assuming chunk_outputs is a list of Path objects
        # This part of the original code was not provided, so it's a placeholder
        # In a real scenario, you'd iterate through chunk_outputs and concatenate frames
        # For simplicity, let's assume a single chunk for now or that chunk_outputs
        # contains the paths to the processed frames.
        # If chunk_outputs contains paths to processed frames, you'd need to read them
        # and concatenate them.

        # Example for a single chunk (if chunk_outputs is a list of paths)
        # This part of the original code was not provided, so it's a placeholder
        # If chunk_outputs is a list of paths to processed frames:
        # frames_to_concatenate = []
        # for chunk_output in chunk_outputs:
        #     for frame_path in chunk_output.iterdir():
        #         if frame_path.name.endswith(".jpg"): # Assuming processed frames are .jpg
        #             frames_to_concatenate.append(cv2.imread(str(frame_path)))
        #             os.remove(str(frame_path)) # Clean up processed frames

        # If chunk_outputs is a list of paths to temporary directories:
        # This part of the original code was not provided, so it's a placeholder
        # If chunk_outputs is a list of paths to temporary directories:
        # frames_to_concatenate = []
        # for chunk_output in chunk_outputs:
        #     for frame_path in chunk_output.iterdir():
        #         if frame_path.name.endswith(".jpg"): # Assuming processed frames are .jpg
        #             frames_to_concatenate.append(cv2.imread(str(frame_path)))
        #             os.remove(str(frame_path)) # Clean up processed frames

        # For now, let's assume a placeholder for concatenation
        # In a real scenario, you'd use a video writer to write frames to a video file
        # This part of the original code was not provided, so it's a placeholder
        # For demonstration, let's just return a placeholder path
        return final_output_path


# Global configuration instance
config = AppConfig.load_from_file()
