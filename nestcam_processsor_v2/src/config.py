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
from processors.video_processor import ProcessingResult  # Add this import

# Load environment variables
load_dotenv()

# GPU Support (CUDA for 4070ti)
import platform

IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# GPU Backend Detection
GPU_BACKEND = "none"
HAS_CUDA = False
HAS_METAL = False
HAS_CUPY = False

try:
    import torch

    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        HAS_CUDA = True
        GPU_BACKEND = "cuda"
        print("✅ NVIDIA GPU detected (CUDA)")
        print(f"   📊 CUDA Version: {torch.version.cuda}")
        print(f"   🎯 GPU Device: {torch.cuda.get_device_name(0)}")

    # Enhanced Metal detection for Mac
    elif IS_MAC:
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                HAS_METAL = True
                GPU_BACKEND = "metal"
                print("✅ Apple Silicon GPU detected (Metal)")
                print("   🍎 Metal Performance Shaders enabled")
                print("   🚀 GPU acceleration available")

                # Test Metal device creation
                device = torch.device("mps")
                test_tensor = torch.randn(100, 100, device=device)
                del test_tensor
                print("   ✅ Metal device test passed")

            else:
                print("⚠️ Metal not available, falling back to CPU")
                print(
                    "   💡 Try: pip install --upgrade torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cpu"
                )
                GPU_BACKEND = "cpu"
        except Exception as e:
            print(f"⚠️ Metal initialization failed: {e}")
            print("   💡 Falling back to CPU")
            GPU_BACKEND = "cpu"

    else:
        GPU_BACKEND = "cpu"
        print("⚠️ No GPU acceleration available, using CPU")

except ImportError as e:
    print(f"⚠️ PyTorch not available for GPU detection: {e}")
    print("   💡 Install with: pip install torch torchvision")
    GPU_BACKEND = "cpu"

# Legacy cupy support for OpenCV CUDA operations
try:
    import cupy as cp

    HAS_CUPY = True
    if not HAS_CUDA and not HAS_METAL:
        GPU_BACKEND = "cuda-legacy"
        print("✅ Legacy CUDA support available (cupy)")
except ImportError:
    print("⚠️ cupy not available for legacy CUDA support")

# Set final GPU availability
HAS_GPU = HAS_CUDA or HAS_METAL or HAS_CUPY

print(f"🎯 GPU Backend: {GPU_BACKEND}")
print(f"🚀 GPU Acceleration: {'Enabled' if HAS_GPU else 'Disabled'}")


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

    # Detailed Analysis Settings
    use_detailed_analysis: bool = Field(
        default=True, description="Enable detailed motion analysis (Pass 2)"
    )
    detail_level: str = Field(
        default="normal",
        pattern="^(light|normal|detailed)$",
        description="Level of detail for motion analysis",
    )
    context_window_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of frames around detected motion to analyze",
    )
    analysis_methods: list = Field(
        default=["white_threshold", "motion_diff"],
        description="Which analysis methods to use",
    )

    processing_state_dir: Optional[Path] = Field(
        default=None, description="Directory to save processing states for resume"
    )
    enable_resume: bool = Field(
        default=True, description="Enable processing resume functionality"
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
    enable_audio: bool = Field(default=True, description="Enable audio processing")
    selected_music: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Selected music for each duration"
    )


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
    ) -> "ProcessingResult":  # Removed -> "ProcessingResult" type hint
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

                # Initialize motion detection variables
                frame_indices = []
                motion_scores = []

                # Initialize baseline variables for streaming
                baseline_avg = None
                baseline_count = 0

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
        """Enable GPU acceleration based on detected backend"""
        backend = detect_gpu_backend()

        if backend == "cuda":
            print("🎯 NVIDIA GPU acceleration enabled (CUDA)")
            self._setup_cuda()
        elif backend == "metal":
            print("🎯 Apple Silicon GPU acceleration enabled (Metal)")
            self._setup_metal()
        elif backend == "cuda-legacy":
            print("🎯 Legacy CUDA acceleration enabled (cupy)")
            self._setup_cuda_legacy()
        else:
            print("⚠️ Using CPU processing")

    def _setup_cuda(self):
        """Setup NVIDIA CUDA acceleration"""
        try:
            import torch

            self.device = torch.device("cuda")
            print(f"📊 CUDA Device: {torch.cuda.get_device_name(0)}")
            print(
                f"🧠 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB"
            )
        except Exception as e:
            print(f"❌ CUDA setup failed: {e}")

    def _setup_metal(self):
        """Setup Apple Metal acceleration"""
        try:
            import torch

            self.device = torch.device("mps")
            print("🍎 Metal acceleration enabled")
            print(f"🧠 Metal Memory: Available on device")
        except Exception as e:
            print(f"❌ Metal setup failed: {e}")

    def _setup_cuda_legacy(self):
        """Setup legacy CUDA with cupy"""
        try:
            import cupy as cp
            import cv2

            # Test CUDA support
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cv2.cuda.setDevice(0)
                print("🔥 Legacy CUDA enabled for OpenCV operations")
            else:
                print("⚠️ CUDA not available for OpenCV operations")
        except Exception as e:
            print(f"❌ Legacy CUDA setup failed: {e}")

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
        """GPU-accelerated frame processing with multiple backends"""
        try:
            if self.gpu_backend == "cuda" and self.device:
                return self._process_frame_torch_cuda(frame)
            elif self.gpu_backend == "metal" and self.device:
                return self._process_frame_torch_metal(frame)
            elif self.gpu_backend == "cuda-legacy":
                return self._process_frame_opencv_cuda(frame)
            else:
                return self._process_frame_cpu(frame)
        except Exception as e:
            print(f"⚠️ GPU processing failed, falling back to CPU: {e}")
            return self._process_frame_cpu(frame)

    def _process_frame_torch_cuda(self, frame):
        """Process frame using PyTorch CUDA"""
        import torch
        import torchvision.transforms as transforms

        # Convert to tensor and move to GPU
        transform = transforms.ToTensor()
        tensor = transform(frame).unsqueeze(0).to(self.device)

        # Apply GPU-based enhancement
        # Example: brightness/contrast adjustment
        enhanced_tensor = torch.clamp(tensor * 1.1, 0, 1)

        # Convert back to numpy
        enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
        enhanced = (enhanced * 255).astype(np.uint8).transpose(1, 2, 0)

        return enhanced

    def _process_frame_torch_metal(self, frame):
        """Process frame using PyTorch Metal (Mac)"""
        import torch
        import torchvision.transforms as transforms

        # Convert to tensor and move to Metal GPU
        transform = transforms.ToTensor()
        tensor = transform(frame).unsqueeze(0).to(self.device)

        # Apply Metal-based enhancement
        enhanced_tensor = torch.clamp(tensor * 1.1, 0, 1)

        # Convert back to numpy
        enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
        enhanced = (enhanced * 255).astype(np.uint8).transpose(1, 2, 0)

        return enhanced

    def _process_frame_opencv_cuda(self, frame):
        """Process frame using OpenCV CUDA (legacy)"""
        import cv2

        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Apply GPU operations
        gpu_frame = cv2.cuda.bilateralFilter(gpu_frame, 9, 75, 75)
        gpu_frame = cv2.cuda.GaussianBlur(gpu_frame, (5, 5), 0)

        # Download from GPU
        return gpu_frame.download()

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
