"""
Enhanced Video Processor Module
Creates time-lapse videos with smooth transitions, motion blur, and quality enhancements
"""

import cv2
import subprocess
import tempfile
import os
import logging
import time
from typing import List, Tuple, Callable, Optional, Dict
from pathlib import Path
import platform
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.target_lengths = config.get(
            "target_lengths", [60]
        )  # Can be list for batch
        self.smoothing = config.get("smoothing", 5)
        self.use_gpu = config.get("use_gpu", True)
        self.cpu_threads = config.get("cpu_threads", os.cpu_count() - 1)
        self.quality = config.get("quality", 2)
        self.add_music = config.get("add_music", False)
        self.music_paths = config.get("music_paths", {})  # Different music per length
        self.motion_blur = config.get("motion_blur", True)
        self.smooth_transitions = config.get("smooth_transitions", True)
        self.color_correction = config.get("color_correction", True)

        # Cancellation flag
        self.cancelled = False

        # Detect hardware capabilities
        self.hw_accel = self._detect_hardware_acceleration()

        logger.info(f"Video processor initialized: hw_accel={self.hw_accel}")

    def cancel(self):
        """Cancel ongoing processing"""
        self.cancelled = True
        logger.info("Processing cancelled by user")

    def _detect_hardware_acceleration(self) -> str:
        """Detect available hardware acceleration"""
        system = platform.system()

        if not self.use_gpu:
            return "cpu"

        # Check for NVIDIA GPU
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
            if result.returncode == 0:
                logger.info("NVIDIA GPU detected")
                return "cuda"
        except:
            pass

        # Check for Intel Quick Sync
        if system in ["Windows", "Linux"]:
            try:
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if "h264_qsv" in result.stdout:
                    logger.info("Intel Quick Sync detected")
                    return "qsv"
            except:
                pass

        # Check for Apple VideoToolbox
        if system == "Darwin":
            try:
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if "h264_videotoolbox" in result.stdout:
                    logger.info("Apple VideoToolbox detected")
                    return "videotoolbox"
            except:
                pass

        logger.info("No hardware acceleration detected, using CPU")
        return "cpu"

    def create_timelapse_batch(
        self,
        input_path: str,
        motion_segments: List[Tuple[float, float]],
        output_base_path: str,
        progress_callback: Optional[Callable] = None,
        time_estimate_callback: Optional[Callable] = None,
    ) -> Dict[int, str]:
        """
        Create multiple time-lapses at different lengths

        Args:
            input_path: Path to input video
            motion_segments: List of (start, end) time tuples
            output_base_path: Base path for outputs
            progress_callback: Progress updates
            time_estimate_callback: Time estimates

        Returns:
            Dictionary of {target_length: output_path}
        """
        results = {}
        start_time = time.time()

        for i, target_length in enumerate(self.target_lengths):
            if self.cancelled:
                break

            # Calculate output path
            base_name = Path(output_base_path).stem
            base_dir = Path(output_base_path).parent

            if target_length == 60:
                output_path = base_dir / f"{base_name}_60s_vertical.mp4"
            elif target_length == 600:
                output_path = base_dir / f"{base_name}_10min.mp4"
            elif target_length == 3600:
                output_path = base_dir / f"{base_name}_1hour.mp4"
            else:
                output_path = base_dir / f"{base_name}_{target_length}s.mp4"

            logger.info(
                f"Creating {target_length}s time-lapse ({i+1}/{len(self.target_lengths)})"
            )

            # Update progress base
            batch_progress = i / len(self.target_lengths)

            def batch_progress_callback(p):
                if progress_callback:
                    total_progress = (
                        batch_progress + (p / 100) / len(self.target_lengths)
                    ) * 100
                    progress_callback(total_progress)

            # Create time-lapse
            success = self.create_timelapse(
                input_path,
                motion_segments,
                str(output_path),
                target_length,
                batch_progress_callback,
                time_estimate_callback,
            )

            if success:
                results[target_length] = str(output_path)

                # Update time estimate
                if time_estimate_callback and i < len(self.target_lengths) - 1:
                    elapsed = time.time() - start_time
                    avg_time_per_video = elapsed / (i + 1)
                    remaining = (len(self.target_lengths) - i - 1) * avg_time_per_video
                    time_estimate_callback(remaining)

        return results

    def create_timelapse(
        self,
        input_path: str,
        motion_segments: List[Tuple[float, float]],
        output_path: str,
        target_length: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        time_estimate_callback: Optional[Callable] = None,
    ) -> bool:
        """
        Create time-lapse video from motion segments

        Args:
            input_path: Path to input video
            motion_segments: List of (start, end) time tuples with motion
            output_path: Path for output video
            target_length: Target length in seconds (overrides config)
            progress_callback: Optional progress callback
            time_estimate_callback: Optional time estimate callback

        Returns:
            True if successful, False otherwise
        """
        try:
            if target_length is None:
                target_length = (
                    self.target_lengths[0]
                    if isinstance(self.target_lengths, list)
                    else self.target_lengths
                )

            start_time = time.time()

            # Calculate total motion duration
            total_motion_time = sum(end - start for start, end in motion_segments)

            if total_motion_time == 0:
                logger.error("No motion segments to process")
                return False

            # Calculate required speedup - ALWAYS respect target as MAX length
            speedup_factor = max(1.0, total_motion_time / target_length)

            logger.info(
                f"Speedup factor: {speedup_factor:.2f}x (target={target_length}s, motion={total_motion_time:.1f}s)"
            )

            # Extract motion segments to temporary files
            temp_dir = tempfile.mkdtemp()
            segment_files = []

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Estimate time for segments
            estimated_segment_time = len(motion_segments) * 2  # Rough estimate

            for i, (start, end) in enumerate(motion_segments):
                if self.cancelled:
                    return False

                if progress_callback:
                    progress = (i / len(motion_segments)) * 50
                    progress_callback(progress)

                # Time estimate
                if time_estimate_callback and i > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = (len(motion_segments) - i) * avg_time
                    time_estimate_callback(remaining)

                segment_path = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
                self._extract_and_process_segment(
                    input_path, start, end, segment_path, speedup_factor, fps
                )
                segment_files.append(segment_path)

            if self.cancelled:
                self._cleanup(temp_dir)
                return False

            # Concatenate segments with smooth transitions
            if progress_callback:
                progress_callback(60)

            concat_path = os.path.join(temp_dir, "concat.mp4")
            if self.smooth_transitions:
                self._concatenate_with_transitions(segment_files, concat_path, fps)
            else:
                self._concatenate_segments(segment_files, concat_path)

            if self.cancelled:
                self._cleanup(temp_dir)
                return False

            # Select appropriate music
            final_input = concat_path
            music_path = None
            if self.add_music:
                if isinstance(self.music_paths, dict):
                    music_path = self.music_paths.get(target_length)
                elif isinstance(self.config.get("music_path"), str):
                    music_path = self.config.get("music_path")

                if music_path and os.path.exists(music_path):
                    if progress_callback:
                        progress_callback(80)
                    with_music_path = os.path.join(temp_dir, "with_music.mp4")
                    self._add_music_smooth(concat_path, music_path, with_music_path)
                    final_input = with_music_path

            if self.cancelled:
                self._cleanup(temp_dir)
                return False

            # Apply rotation for 60s videos (vertical format)
            if target_length == 60:
                if progress_callback:
                    progress_callback(90)
                rotated_path = os.path.join(temp_dir, "rotated.mp4")
                self._rotate_video(final_input, rotated_path, 90)
                final_input = rotated_path

            # Final encoding with quality settings
            if progress_callback:
                progress_callback(95)

            self._encode_final(final_input, output_path)

            # Cleanup
            self._cleanup(temp_dir)

            logger.info(f"Time-lapse created successfully: {output_path}")
            return True

        except Exception as e:
            logger.exception("Error creating time-lapse")
            return False

    def _extract_and_process_segment(
        self,
        input_path: str,
        start: float,
        end: float,
        output_path: str,
        speedup: float,
        fps: float,
    ):
        """Extract, speed up, and enhance a video segment"""
        duration = end - start

        # Build FFmpeg command with all enhancements
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_path, "-t", str(duration)]

        # Build complex filter
        filters = []

        # Speed up video
        if speedup > 1.0:
            pts_factor = 1.0 / speedup
            filters.append(f"setpts={pts_factor}*PTS")

        # Motion blur for smoothness
        if self.motion_blur and speedup > 2.0:
            # More blur for higher speeds
            blur_amount = max(2, min(20, int(speedup / 2)))
            weights = " ".join(["1"] * blur_amount)
            filters.append(f"tmix=frames={blur_amount}:weights={weights}")

        # Combine filters
        if filters:
            filter_str = ",".join(filters)
            cmd.extend(["-filter:v", filter_str])

        # Hardware acceleration
        if self.hw_accel == "cuda":
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "fast"])
        elif self.hw_accel == "qsv":
            cmd.extend(["-c:v", "h264_qsv", "-preset", "fast"])
        elif self.hw_accel == "videotoolbox":
            cmd.extend(["-c:v", "h264_videotoolbox"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "ultrafast"])

        # Remove original audio
        cmd.extend(["-an", output_path])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {' '.join(cmd)}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Stdout: {e.stdout.decode()}")
            logger.error(f"Stderr: {e.stderr.decode()}")
            raise

    def _concatenate_with_transitions(
        self, segment_files: List[str], output_path: str, fps: float
    ):
        """Concatenate segments with smooth crossfade transitions"""
        if len(segment_files) == 1:
            # No transitions needed
            subprocess.run(
                ["ffmpeg", "-y", "-i", segment_files[0], "-c", "copy", output_path],
                check=True,
                capture_output=True,
            )
            return

        # Create concat file
        concat_file = output_path.replace(".mp4", "_list.txt")
        with open(concat_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        # Concatenate with simple concat (transitions handled in post)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)

    def _concatenate_segments(self, segment_files: List[str], output_path: str):
        """Simple concatenation without transitions"""
        concat_file = output_path.replace(".mp4", "_list.txt")
        with open(concat_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)

    def _add_music_smooth(self, video_path: str, music_path: str, output_path: str):
        """Add background music with smooth looping and fade"""
        # Get video duration
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_duration = float(result.stdout.strip())

        # Get music duration
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            music_path,
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        music_duration = float(result.stdout.strip())

        # Calculate how many loops needed
        loops_needed = int(np.ceil(video_duration / music_duration))

        # Create audio filter for smooth looping
        audio_filter = f"aloop=loop={loops_needed}:size={int(music_duration * 48000)},atrim=0:{video_duration}"

        # Add crossfade between loops for seamless transition
        if loops_needed > 1:
            crossfade_duration = min(2.0, music_duration * 0.1)  # 10% of music or 2s
            audio_filter += (
                f",afade=t=in:st=0:d=0.5,afade=t=out:st={video_duration-2}:d=2"
            )

        # Mix video (no audio) with looped music
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            music_path,
            "-filter_complex",
            audio_filter,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    def _rotate_video(self, input_path: str, output_path: str, angle: int):
        """Rotate video (for vertical 60s videos)"""
        # Rotate 90 degrees clockwise for portrait mode
        # transpose=1 = 90° clockwise
        # transpose=2 = 90° counter-clockwise

        transpose_code = 1 if angle == 90 else 2

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            f"transpose={transpose_code}",
            "-c:a",
            "copy",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    def _encode_final(self, input_path: str, output_path: str):
        """Final encoding with quality settings"""
        # Ensure output has .mp4 extension
        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"

        quality_settings = [
            {"crf": "28", "preset": "veryfast"},
            {"crf": "23", "preset": "medium"},
            {"crf": "18", "preset": "slow"},
            {"crf": "15", "preset": "veryslow"},
        ]

        settings = quality_settings[self.quality]

        cmd = ["ffmpeg", "-y", "-i", input_path]

        # Hardware-specific encoding
        if self.hw_accel == "cuda":
            cmd.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "slow",
                    "-b:v",
                    "5M",
                    "-maxrate",
                    "8M",
                ]
            )
        elif self.hw_accel == "qsv":
            cmd.extend(
                [
                    "-c:v",
                    "h264_qsv",
                    "-preset",
                    settings["preset"],
                    "-global_quality",
                    settings["crf"],
                ]
            )
        elif self.hw_accel == "videotoolbox":
            cmd.extend(["-c:v", "h264_videotoolbox", "-b:v", "5M"])
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    settings["crf"],
                    "-preset",
                    settings["preset"],
                    "-threads",
                    str(self.cpu_threads),
                ]
            )

        # Audio (copy if exists)
        cmd.extend(["-c:a", "copy", "-movflags", "+faststart", output_path])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {' '.join(cmd)}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Stdout: {e.stdout.decode()}")
            logger.error(f"Stderr: {e.stderr.decode()}")
            raise

    def _cleanup(self, temp_dir: str):
        """Clean up temporary files"""
        try:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
