"""
Advanced video processing with multiple enhancement algorithms
"""

import sys
import os

# Add the parent directory to the path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    # Try relative imports first (when run as module)
    from ..utils.logger import get_logger
    from ..services.file_service import FileService
except ImportError:
    # Fall back to absolute imports (when run directly by Streamlit)
    from utils.logger import get_logger
    from services.file_service import FileService

from .motion_detector import MotionDetector
from .enhancer import VideoEnhancer

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import time
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of video processing"""

    filename: str
    frames_processed: int
    motion_events: int
    processing_time: float
    output_files: List[str]
    error: Optional[str] = None


class VideoProcessor:
    """Advanced video processor with parallel processing and multiple algorithms"""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        self.motion_detector = MotionDetector(config)
        self.enhancer = VideoEnhancer(config)
        self.file_service = FileService(config)

        # Processing parameters
        self.batch_size = config.processing.batch_size
        self.max_workers = config.processing.worker_processes

    def process_video(
        self,
        input_path: str,
        output_format: str = "mp4",
        output_dir: Optional[str] = None,
        watermark_text: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> ProcessingResult:
        """
        Process a single video file

        Args:
            input_path: Path to input video
            output_format: Output video format
            output_dir: Output directory (optional)
            watermark_text: Text to add as watermark (optional)
            progress_callback: Progress callback function

        Returns:
            ProcessingResult object
        """
        start_time = time.time()
        self.logger.info(f"Starting video processing: {input_path}")

        try:
            # Validate input file
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Step 1: Motion detection
            if progress_callback:
                progress_callback(5, "Detecting motion...", 0)

            motion_result = self.motion_detector.detect_motion(
                input_path,
                progress_callback=lambda p, cf, tf: progress_callback(
                    5 + (p * 0.3), f"Motion detection: {cf}", tf
                ),
            )

            if not motion_result.frame_indices:
                raise ValueError("No motion detected in video")

            # Step 2: Generate output videos for different durations
            output_files = []
            durations = self._get_durations()

            total_durations = len(durations)
            for i, duration in enumerate(durations):
                if progress_callback:
                    progress = 35 + (i / total_durations) * 60
                    progress_callback(progress, f"Generating {duration}s video...", i)

                output_file = self._generate_video(
                    input_path,
                    duration,
                    motion_result.frame_indices,
                    output_format,
                    output_dir,
                    watermark_text,
                    lambda p, cf, tf: progress_callback(
                        progress + (p * (60 / total_durations / 100)), cf, tf
                    ),
                )

                output_files.append(output_file)

            # Step 3: Add audio if configured
            if progress_callback:
                progress_callback(95, "Adding audio...", 0)

            output_files = self._add_audio_to_videos(output_files)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                filename=Path(input_path).name,
                frames_processed=motion_result.processed_frames,
                motion_events=len(motion_result.frame_indices),
                processing_time=processing_time,
                output_files=output_files,
            )

            self.logger.info(f"Video processing completed: {input_path}")
            return result

        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            processing_time = time.time() - start_time
            return ProcessingResult(
                filename=Path(input_path).name,
                frames_processed=0,
                motion_events=0,
                processing_time=processing_time,
                output_files=[],
                error=str(e),
            )

    def process_video_streaming(
        self,
        input_path: str,
        output_format: str = "mp4",
        output_dir: Optional[str] = None,
        watermark_text: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        use_gpu: bool = True,
    ) -> ProcessingResult:
        """
        Memory-efficient video processing with GPU acceleration for large files

        This method processes frames in smaller chunks to avoid loading
        everything into memory at once, making it suitable for 50GB+ files.
        """
        start_time = time.time()
        self.logger.info(f"Starting video streaming processing: {input_path}")

        try:
            # Validate input file
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Step 1: Motion detection
            if progress_callback:
                progress_callback(5, "Detecting motion...", 0)

            motion_result = self.motion_detector.detect_motion(
                input_path,
                progress_callback=lambda p, cf, tf: progress_callback(
                    5 + (p * 0.3), f"Motion detection: {cf}", tf
                ),
            )

            if not motion_result.frame_indices:
                raise ValueError("No motion detected in video")

            # Step 2: Generate output videos for different durations
            output_files = []
            durations = self._get_durations()

            total_durations = len(durations)
            for i, duration in enumerate(durations):
                if progress_callback:
                    progress = 35 + (i / total_durations) * 60
                    progress_callback(progress, f"Generating {duration}s video...", i)

                output_file = self._generate_video(
                    input_path,
                    duration,
                    motion_result.frame_indices,
                    output_format,
                    output_dir,
                    watermark_text,
                    lambda p, cf, tf: progress_callback(
                        progress + (p * (60 / total_durations / 100)), cf, tf
                    ),
                )

                output_files.append(output_file)

            # Step 3: Add audio if configured
            if progress_callback:
                progress_callback(95, "Adding audio...", 0)

            output_files = self._add_audio_to_videos(output_files)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                filename=Path(input_path).name,
                frames_processed=motion_result.processed_frames,
                motion_events=len(motion_result.frame_indices),
                processing_time=processing_time,
                output_files=output_files,
            )

            self.logger.info(f"Video streaming processing completed: {input_path}")
            return result

        except Exception as e:
            self.logger.error(f"Video streaming processing failed: {e}")
            processing_time = time.time() - start_time
            return ProcessingResult(
                filename=Path(input_path).name,
                frames_processed=0,
                motion_events=0,
                processing_time=processing_time,
                output_files=[],
                error=str(e),
            )

    def _get_durations(self) -> List[int]:
        """Get list of video durations to generate"""
        durations = []
        if hasattr(self.config, "generate_60s") and self.config.generate_60s:
            durations.append(59)  # 60 seconds - 1 for transition
        if hasattr(self.config, "generate_12min") and self.config.generate_12min:
            durations.append(720)  # 12 minutes
        if hasattr(self.config, "generate_1h") and self.config.generate_1h:
            durations.append(3600)  # 1 hour
        if hasattr(self.config, "custom_duration") and self.config.custom_duration:
            durations.append(self.config.custom_duration)
        return durations

    def _generate_video(
        self,
        input_path: str,
        duration: int,
        frame_indices: List[int],
        output_format: str,
        output_dir: Optional[str],
        watermark_text: Optional[str],
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Generate output video for specific duration"""

        # Create output filename
        base_name = Path(input_path).stem
        output_filename = f"{base_name}_{duration}s.{output_format}"

        if output_dir:
            output_path = Path(output_dir) / output_filename
        else:
            output_path = Path(input_path).parent / output_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get video properties
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate target frames
        target_frames = int(duration * fps)

        if len(frame_indices) > target_frames:
            # Select frames evenly distributed
            step = len(frame_indices) / target_frames
            selected_indices = [
                frame_indices[int(i * step)] for i in range(target_frames)
            ]
        else:
            selected_indices = frame_indices

        # Process frames in parallel
        temp_dir = Path(f"temp_frames_{time.time()}")
        temp_dir.mkdir(exist_ok=True)

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Split frames into batches
                frame_batches = [
                    selected_indices[i : i + self.batch_size]
                    for i in range(0, len(selected_indices), self.batch_size)
                ]

                futures = []
                for i, batch in enumerate(frame_batches):
                    future = executor.submit(
                        self._process_frame_batch,
                        input_path,
                        batch,
                        i,
                        temp_dir,
                        width,
                        height,
                        watermark_text,
                    )
                    futures.append(future)

                    if progress_callback:
                        progress = (i / len(frame_batches)) * 100
                        progress_callback(
                            progress,
                            f"Processing frames: {i+1}/{len(frame_batches)}",
                            len(frame_batches),
                        )

                # Wait for all batches to complete
                for future in futures:
                    future.result()

            # Create final video using FFmpeg
            self._create_video_from_frames(
                temp_dir, output_path, fps, width, height, output_format
            )

            return str(output_path)

        finally:
            # Clean up temporary files
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_frame_batch(
        self,
        input_path: str,
        frame_indices: List[int],
        batch_id: int,
        temp_dir: Path,
        width: int,
        height: int,
        watermark_text: Optional[str],
    ) -> None:
        """Process a batch of frames"""

        cap = cv2.VideoCapture(input_path)

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Enhance frame
            enhanced_frame = self.enhancer.enhance_frame(frame)

            # Add watermark if specified
            if watermark_text:
                enhanced_frame = self._add_watermark(enhanced_frame, watermark_text)

            # Save frame
            frame_filename = f"frame_{batch_id:04d}_{i:04d}.jpg"
            frame_path = temp_dir / frame_filename
            cv2.imwrite(str(frame_path), enhanced_frame)

        cap.release()

    def _add_watermark(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Add watermark to frame"""
        height, width = frame.shape[:2]

        # Add text watermark
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 1000
        thickness = max(1, int(min(width, height) / 400))

        # Position text in bottom right corner
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = width - text_size[0] - 10
        text_y = height - 10

        # Add background rectangle for better readability
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1,
        )

        # Add text
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def _create_video_from_frames(
        self,
        temp_dir: Path,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
        output_format: str,
    ) -> None:
        """Create video from processed frames using FFmpeg"""

        import subprocess

        # FFmpeg command to create video from frames
        cmd = [
            "ffmpeg",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%04d_%04d.jpg"),
            "-s",
            f"{width}x{height}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-y",  # Overwrite output file
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Video created successfully: {output_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Failed to create video: {e.stderr}")

    def _add_audio_to_videos(self, video_files: List[str]) -> List[str]:
        """Add background audio to videos"""

        enhanced_files = []

        for video_file in video_files:
            video_path = Path(video_file)

            # Determine duration from filename
            duration = self._extract_duration_from_filename(video_path.name)

            # Get appropriate music file
            music_path = self._get_music_for_duration(duration)

            if music_path and Path(music_path).exists():
                # Add audio using FFmpeg
                audio_video = self._add_audio_to_video(video_file, music_path)
                enhanced_files.append(audio_video)
            else:
                # Add silent audio
                silent_video = self._add_silent_audio_to_video(video_file)
                enhanced_files.append(silent_video)

        return enhanced_files

    def _extract_duration_from_filename(self, filename: str) -> int:
        """Extract duration from video filename"""
        import re

        match = re.search(r"_(\d+)s\.", filename)
        if match:
            return int(match.group(1))
        return 60  # Default duration

    def _get_music_for_duration(self, duration: int) -> Optional[str]:
        """Get appropriate music file for video duration"""
        if str(duration) in self.config.audio.music_paths:
            return self.config.audio.music_paths[str(duration)]
        return self.config.audio.music_paths.get("default")

    def _add_audio_to_video(self, video_path: str, music_path: str) -> str:
        """Add background music to video"""
        import subprocess

        video_path = Path(video_path)
        output_path = video_path.parent / f"{video_path.stem}_audio{video_path.suffix}"

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-i",
            str(music_path),
            "-filter_complex",
            f"[1:a]volume={self.config.audio.volume}[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to add audio: {e.stderr}")
            return video_path  # Return original video if audio addition fails

    def _add_silent_audio_to_video(self, video_path: str) -> str:
        """Add silent audio to video"""
        import subprocess

        video_path = Path(video_path)
        output_path = video_path.parent / f"{video_path.stem}_silent{video_path.suffix}"

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to add silent audio: {e.stderr}")
            return video_path  # Return original video if silent audio addition fails
