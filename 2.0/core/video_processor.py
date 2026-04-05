import cv2
import ffmpeg
import math
import numpy as np
from typing import List, Tuple, Optional
import os
from .motion_detector import MotionDetector
from utils.logger import logger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class VideoProcessor:
    def __init__(self, motion_detector: MotionDetector):
        self.motion_detector = motion_detector
        self.target_brightness = None
        self.brightness_roi = (
            0.25,
            0.25,
            0.5,
            0.5,
        )  # relative center ROI for nest exposure

    def _is_bad_frame(
        self,
        frame: np.ndarray,
        black_thresh: int = 20,
        white_thresh: int = 235,
        uniformity_thresh: int = 8,
    ) -> bool:
        if frame is None:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        if mean_val < black_thresh or mean_val > white_thresh:
            return True
        if std_val < uniformity_thresh:
            return True
        return False

    def _get_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        rx, ry, rw, rh = self.brightness_roi
        x = int(w * rx)
        y = int(h * ry)
        return x, y, int(w * rw), int(h * rh)

    def _get_brightness(self, frame: np.ndarray) -> float:
        x, y, w, h = self._get_roi(frame)
        roi = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _sample_frame_brightness(self, input_path: str, timestamp: float) -> float:
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return 128.0
        return self._get_brightness(frame)

    def _sample_segment_brightness(
        self, input_path: str, start: float, end: float, sample_count: int = 3
    ) -> float:
        if end <= start:
            return self._sample_frame_brightness(input_path, start)

        cap = cv2.VideoCapture(input_path)
        sample_times = np.linspace(start, end, min(sample_count, 3))
        brightness_samples = []

        for timestamp in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            if self._is_bad_frame(frame):
                continue
            brightness_samples.append(self._get_brightness(frame))

        cap.release()

        if brightness_samples:
            return float(np.median(brightness_samples))
        return self._sample_frame_brightness(input_path, start)

    def _brightness_adjustment(self, current: float) -> float:
        if self.target_brightness is None or current <= 0:
            return 0.0
        delta = (self.target_brightness - current) / 255.0
        return float(np.clip(delta, -0.3, 0.3))

    def extract_motion_segments(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None,
        cancel_flag=None,
        min_segment_duration: float = 0.5,
        frame_subsample: int = 5,
        motion_buffer: float = 2.0,
        merge_threshold: float = 3.0,
        use_tqdm: bool = False,
    ) -> List[Tuple[float, float]]:
        """Extract segments with motion using optimized detection"""
        probe = ffmpeg.probe(input_path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        duration = float(video_info["duration"])
        fps = eval(video_info["r_frame_rate"])

        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(round(duration * fps))
        frame_interval = 1.0 / fps

        effective_total = max(1, math.ceil(frame_count / frame_subsample))
        pbar = None
        if use_tqdm and tqdm is not None and progress_callback is None:
            pbar = tqdm(
                total=effective_total,
                desc="Motion detection",
                unit="frames",
                ncols=80,
                smoothing=0.1,
            )

        motion_segments = []
        current_segment = None
        last_motion_end = 0
        bad_frame_count = 0
        brightness_values = []

        logger.info(
            f"Starting motion detection: {frame_count} frames, {duration:.1f}s, subsampling every {frame_subsample} frames"
        )

        for i in range(0, frame_count, frame_subsample):
            if cancel_flag and cancel_flag.cancel_flag:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            if self._is_bad_frame(frame):
                bad_frame_count += 1
                if current_segment:
                    segment_duration = current_segment[1] - current_segment[0]
                    if segment_duration >= min_segment_duration:
                        motion_segments.append(tuple(current_segment))
                    last_motion_end = current_segment[1]
                    current_segment = None
                if pbar:
                    pbar.update(1)
                continue

            brightness_values.append(self._get_brightness(frame))
            has_motion, _ = self.motion_detector.detect_motion(frame, preprocess=True)
            timestamp = i / fps

            if has_motion:
                if current_segment is None:
                    # Start new segment with buffer
                    start_time = max(0, timestamp - motion_buffer)
                    current_segment = [start_time, timestamp + motion_buffer]
                else:
                    # Extend current segment
                    current_segment[1] = timestamp + motion_buffer
            else:
                if current_segment:
                    # Check if we should merge with previous segment
                    if (
                        last_motion_end > 0
                        and (current_segment[0] - last_motion_end) < merge_threshold
                    ):
                        # Merge segments
                        current_segment[0] = last_motion_end
                        # Remove the last segment and extend the previous one
                        if motion_segments:
                            motion_segments[-1] = (
                                motion_segments[-1][0],
                                current_segment[1],
                            )
                        else:
                            motion_segments.append(tuple(current_segment))
                    else:
                        # Ensure segment meets minimum duration
                        segment_duration = current_segment[1] - current_segment[0]
                        if segment_duration >= min_segment_duration:
                            motion_segments.append(tuple(current_segment))

                    last_motion_end = current_segment[1]
                    current_segment = None

            if pbar:
                pbar.update(1)

            # Report progress every 100 sampled frames
            if i % (frame_subsample * 100) == 0 and progress_callback:
                progress_callback(i + 1, frame_count)

        # Handle final segment
        if current_segment:
            segment_duration = current_segment[1] - current_segment[0]
            if segment_duration >= min_segment_duration:
                motion_segments.append(tuple(current_segment))

        if pbar:
            pbar.close()
        cap.release()
        if brightness_values:
            self.target_brightness = float(np.median(brightness_values))
            logger.info(
                f"Normalized target brightness set to {self.target_brightness:.1f}"
            )
        else:
            self.target_brightness = 128.0
            logger.info(
                "No valid brightness samples collected; using default target brightness 128"
            )
        logger.info(f"Skipped {bad_frame_count} bad frames during motion detection")

        # Post-process: ensure no overlapping segments and sort
        motion_segments.sort()
        merged_segments = []
        for start, end in motion_segments:
            if merged_segments and start <= merged_segments[-1][1]:
                # Overlap - extend the previous segment
                merged_segments[-1] = (
                    merged_segments[-1][0],
                    max(merged_segments[-1][1], end),
                )
            else:
                merged_segments.append((start, end))

        logger.info(f"Motion detection complete: {len(merged_segments)} segments found")
        for i, (start, end) in enumerate(merged_segments[:5]):
            logger.info(
                f"  Segment {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)"
            )
        if len(merged_segments) > 5:
            logger.info(f"  ... and {len(merged_segments)-5} more segments")

        return merged_segments

    def create_timelapse(
        self,
        input_path: str,
        motion_segments: List[Tuple[float, float]],
        target_duration: int,
        output_path: str,
        mode: str = "fast",
        quality: str = "high",
        progress_callback=None,
        cancel_flag=None,
        output_format: str = "mp4",
    ):
        """Create time-lapse from motion segments using selected mode"""
        if mode == "fast":
            result = self._create_timelapse_fast(
                input_path,
                motion_segments,
                target_duration,
                output_path,
                quality,
                progress_callback,
                cancel_flag,
                output_format,
            )
        else:
            result = self._create_timelapse_quality(
                input_path,
                motion_segments,
                target_duration,
                output_path,
                quality,
                progress_callback,
                cancel_flag,
                output_format,
            )

        if target_duration == 60:
            return self._rotate_vertical_output(result, output_format, quality)
        return result

    def _build_atempo_filter(self, speed_factor: float) -> str:
        """Build a valid FFmpeg atempo filter chain for any speed factor."""
        if speed_factor <= 0:
            raise ValueError("Speed factor must be greater than 0")

        parts = []
        factor = speed_factor

        while factor > 2.0:
            parts.append("atempo=2.0")
            factor /= 2.0

        while factor < 0.5:
            parts.append("atempo=0.5")
            factor *= 2.0

        # Accept a small floating point remainder as final factor
        if abs(factor - 1.0) > 1e-6:
            parts.append(f"atempo={factor:.6f}")

        return ",".join(parts) if parts else "atempo=1.0"

    def _probe_duration(self, path: str) -> float:
        """Return the duration of a media file in seconds."""
        probe = ffmpeg.probe(path)
        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"), None
        )
        if not video_stream:
            raise ValueError(f"No video stream found in {path}")
        return float(video_stream["duration"])

    def _correct_final_duration(
        self,
        path: str,
        target_duration: float,
        output_format: str = "mp4",
        quality: str = "high",
    ) -> str:
        """Fix the final duration if it deviates from the requested target."""
        actual_duration = self._probe_duration(path)
        tolerance = 0.5
        if abs(actual_duration - target_duration) <= tolerance:
            return path

        speed_adjust = actual_duration / target_duration
        atempo_filter = self._build_atempo_filter(speed_adjust)

        corrected_path = f"{path}.fixed{os.path.splitext(path)[1]}"
        if output_format == "webm":
            vcodec, acodec = "libvpx-vp9", "libopus"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )
        elif output_format == "avi":
            vcodec, acodec = "libxvid", "mp3"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )
        else:
            vcodec, acodec = "libx264", "aac"
            bitrate = None

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-filter:v",
            f"setpts=PTS/{speed_adjust}",
            "-filter:a",
            atempo_filter,
            "-c:v",
            vcodec,
            "-preset",
            "medium",
            "-crf",
            "23",
        ]
        if bitrate:
            cmd += ["-b:v", bitrate]
        cmd += ["-c:a", acodec, corrected_path]

        import subprocess

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Final duration correction failed: {result.stderr[-300:]}")
            return path

        os.replace(corrected_path, path)
        logger.info(
            f"Corrected final duration from {actual_duration:.2f}s to target {target_duration:.2f}s"
        )
        return path

    def _combine_segments_with_crossfade(
        self,
        segment_files: List[str],
        output_path: str,
        output_format: str = "mp4",
        quality: str = "high",
        transition_duration: float = 0.4,
    ) -> str:
        """Concatenate segment files with crossfade transitions."""
        if not segment_files:
            raise ValueError("No segments to concatenate")
        if len(segment_files) == 1:
            os.replace(segment_files[0], output_path)
            return output_path

        durations = [self._probe_duration(path) for path in segment_files]
        vcodec = "libx264"
        acodec = "aac"
        bitrate = None
        if output_format == "webm":
            vcodec, acodec = "libvpx-vp9", "libopus"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )
        elif output_format == "avi":
            vcodec, acodec = "libxvid", "mp3"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )

        filter_chain = []
        for idx in range(len(segment_files)):
            filter_chain.append(f"[{idx}:v]format=yuv420p,setsar=1[v{idx}];")
            filter_chain.append(f"[{idx}:a]aresample=async=1[a{idx}];")

        offset = durations[0] - transition_duration
        filter_chain.append(
            f"[v0][v1]xfade=transition=fade:duration={transition_duration}:offset={offset}[v01];"
        )
        filter_chain.append(f"[a0][a1]acrossfade=d={transition_duration}[a01];")
        last_v = "v01"
        last_a = "a01"

        for idx in range(2, len(segment_files)):
            offset += durations[idx - 1] - transition_duration
            output_v = f"v0{idx}"
            output_a = f"a0{idx}"
            filter_chain.append(
                f"[{last_v}][v{idx}]xfade=transition=fade:duration={transition_duration}:offset={offset}[{output_v}];"
            )
            filter_chain.append(
                f"[{last_a}][a{idx}]acrossfade=d={transition_duration}[{output_a}];"
            )
            last_v = output_v
            last_a = output_a

        filter_complex = "".join(filter_chain)
        cmd = ["ffmpeg", "-y"]
        for seg in segment_files:
            cmd += ["-i", seg]
        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{last_v}]",
            "-map",
            f"[{last_a}]",
            "-c:v",
            vcodec,
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            acodec,
        ]
        if bitrate:
            cmd += ["-b:v", bitrate]
        cmd += ["-b:a", "128k", output_path]

        import subprocess

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Crossfade concat failed: {result.stderr[-500:]}")
            raise RuntimeError(f"Crossfade concat failed: {result.stderr[-500:]}")

        return output_path

    def _apply_deflicker_pass(
        self,
        path: str,
        output_format: str = "mp4",
        quality: str = "high",
    ) -> str:
        """Run a final deflicker pass on the assembled video."""
        ext = os.path.splitext(path)[1]
        deflicker_path = f"{path}.deflicker{ext}"

        if output_format == "webm":
            vcodec, acodec = "libvpx-vp9", "libopus"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )
        elif output_format == "avi":
            vcodec, acodec = "libxvid", "mp3"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )
        else:
            vcodec, acodec = "libx264", "aac"
            bitrate = None

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-vf",
            "deflicker",
            "-c:v",
            vcodec,
            "-preset",
            "medium",
            "-crf",
            "23",
        ]
        if bitrate:
            cmd += ["-b:v", bitrate]
        cmd += ["-c:a", acodec, "-b:a", "128k", deflicker_path]

        import subprocess

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                f"Deflicker pass failed or unsupported: {result.stderr[-300:]}"
            )
            return path

        os.replace(deflicker_path, path)
        logger.info(f"Applied final deflicker pass to {path}")
        return path

    def _finalize_output(
        self,
        path: str,
        target_duration: float,
        output_format: str = "mp4",
        quality: str = "high",
    ) -> str:
        corrected = self._correct_final_duration(
            path, target_duration, output_format, quality
        )
        return self._apply_deflicker_pass(corrected, output_format, quality)

    def _rotate_vertical_output(
        self,
        path: str,
        output_format: str = "mp4",
        quality: str = "high",
    ) -> str:
        """Rotate the final video 90 degrees to the right for vertical output."""
        rotated_path = f"{path}.rotated{os.path.splitext(path)[1]}"
        if output_format == "webm":
            vcodec, acodec = "libvpx-vp9", "libopus"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )
        elif output_format == "avi":
            vcodec, acodec = "libxvid", "mp3"
            bitrate = (
                "2M" if quality == "high" else "1M" if quality == "medium" else "512K"
            )
        else:
            vcodec, acodec = "libx264", "aac"
            bitrate = None

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-vf",
            "transpose=1",
            "-c:v",
            vcodec,
            "-preset",
            "medium",
            "-crf",
            "23",
        ]
        if bitrate:
            cmd += ["-b:v", bitrate]
        cmd += ["-c:a", acodec, rotated_path]

        import subprocess

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Vertical rotation failed: {result.stderr[-300:]}")
            return path

        os.replace(rotated_path, path)
        logger.info(f"Rotated final output to vertical orientation: {path}")
        return path

    def _create_timelapse_fast(
        self,
        input_path: str,
        motion_segments: List[Tuple[float, float]],
        target_duration: int,
        output_path: str,
        quality: str = "high",
        progress_callback=None,
        cancel_flag=None,
        output_format: str = "mp4",
    ):
        """Create time-lapse using FFmpeg complex filters (fast)"""
        import subprocess
        import tempfile

        total_motion_time = sum(end - start for start, end in motion_segments)

        logger.info(f"Fast mode segments: {len(motion_segments)} segments")
        for i, (start, end) in enumerate(motion_segments[:5]):  # Log first 5 segments
            logger.info(f"  Segment {i}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        if len(motion_segments) > 5:
            logger.info(f"  ... and {len(motion_segments)-5} more segments")

        if total_motion_time == 0:
            raise ValueError("No motion detected in video")

        speed_factor = total_motion_time / target_duration
        if speed_factor <= 0:
            raise ValueError("Invalid speed factor calculated from motion segments")

        atempo_filter = self._build_atempo_filter(speed_factor)

        logger.info(
            f"Fast mode: total_motion_time={total_motion_time:.2f}s, target_duration={target_duration}s, speed_factor={speed_factor:.6f}, atempo={atempo_filter}"
        )

        # If we have less motion than target, we need to slow down, not speed up
        if total_motion_time < target_duration:
            logger.warning(
                f"Insufficient motion ({total_motion_time:.1f}s) for target ({target_duration}s). Video will be slower than requested."
            )

        # Set quality parameters
        quality_settings = {
            "high": {"preset": "fast", "crf": "20"},
            "medium": {"preset": "faster", "crf": "25"},
            "low": {"preset": "ultrafast", "crf": "30"},
        }
        settings = quality_settings.get(quality, quality_settings["high"])

        # Create temporary directory for segment files
        temp_dir = tempfile.mkdtemp(prefix="nestcams_fast_")
        try:
            # Update progress: Starting time-lapse creation
            if progress_callback:
                progress_callback(35, 100)

            # Extract segments without speed adjustment first
            segment_files = []
            total_segments = len(motion_segments)

            for idx, (start, end) in enumerate(motion_segments):
                if cancel_flag and cancel_flag.cancel_flag:
                    return output_path

                segment_path = os.path.join(temp_dir, f"segment_{idx:04d}.mp4")

                # Normalize brightness per segment and apply speed adjustment
                brightness = self._sample_segment_brightness(input_path, start, end)
                adjust = self._brightness_adjustment(brightness)
                vf_filter = f"setpts=PTS/{speed_factor}"
                if abs(adjust) > 1e-6:
                    vf_filter += f",eq=brightness={adjust:.4f}"

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-filter:v",
                    vf_filter,
                    "-filter:a",
                    atempo_filter,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "25",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    segment_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(
                        f"Segment {idx} extraction failed: {result.stderr[-200:]}"
                    )
                    continue

                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    segment_files.append(segment_path)

                # Update progress: 35% to 55% during segment extraction
                if progress_callback:
                    extract_progress = 35 + (idx / total_segments) * 20
                    progress_callback(int(extract_progress), 100)

            if not segment_files:
                raise ValueError("No segments were successfully extracted")

            # Update progress: 55% during concatenation
            if progress_callback:
                progress_callback(55, 100)

            output_path = self._combine_segments_with_crossfade(
                segment_files,
                output_path,
                output_format,
                quality,
                transition_duration=0.4,
            )

            if progress_callback:
                progress_callback(70, 100)

            corrected_output = self._finalize_output(
                output_path, float(target_duration), output_format, quality
            )

            logger.info(f"Fast time-lapse created: {corrected_output}")
            return corrected_output

        finally:
            # Clean up temporary files
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _create_timelapse_quality(
        self,
        input_path: str,
        motion_segments: List[Tuple[float, float]],
        target_duration: int,
        output_path: str,
        quality: str = "high",
        progress_callback=None,
        cancel_flag=None,
        output_format: str = "mp4",
    ):
        """Create time-lapse using segment extraction + concat (quality/compatible)"""
        import subprocess
        import tempfile

        total_motion_time = sum(end - start for start, end in motion_segments)

        if total_motion_time == 0:
            raise ValueError("No motion detected in video")

        if total_motion_time < target_duration:
            logger.warning(
                f"Insufficient motion time ({total_motion_time:.1f}s) for target duration ({target_duration}s). "
                f"Output will be slower than target."
            )

        speed_factor = total_motion_time / target_duration

        # Set quality parameters
        quality_settings = {
            "high": {"preset": "slow", "crf": "18"},
            "medium": {"preset": "medium", "crf": "23"},
            "low": {"preset": "fast", "crf": "28"},
        }
        settings = quality_settings.get(quality, quality_settings["high"])

        # Create temporary directory for segment files
        temp_dir = tempfile.mkdtemp(prefix="nestcams_")
        try:
            # Extract segments to temporary files with proper H.264 encoding
            segment_files = []
            total_segments = len(motion_segments)

            for idx, (start, end) in enumerate(motion_segments):
                if cancel_flag and cancel_flag.cancel_flag:
                    return output_path

                segment_path = os.path.join(temp_dir, f"segment_{idx:04d}.mp4")

                # Extract segment with appropriate codec for compatibility
                if output_format == "webm":
                    vcodec, acodec = "libvpx-vp9", "libopus"
                elif output_format == "avi":
                    vcodec, acodec = "libxvid", "mp3"
                else:  # mp4
                    vcodec, acodec = "libx264", "aac"

                audio_filter = self._build_atempo_filter(speed_factor)
                brightness = self._sample_segment_brightness(input_path, start, end)
                adjust = self._brightness_adjustment(brightness)
                vf_filter = f"setpts=PTS/{speed_factor}"
                if abs(adjust) > 1e-6:
                    vf_filter += f",eq=brightness={adjust:.4f}"

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-vf",
                    vf_filter,
                    "-filter:a",
                    audio_filter,
                    "-c:v",
                    vcodec,
                    "-preset",
                    settings["preset"],
                    "-crf",
                    settings["crf"],
                    "-c:a",
                    acodec,
                    segment_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(
                        f"Segment {idx} extraction failed: {result.stderr[-200:]}"
                    )
                    continue  # Skip segments that fail

                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    segment_files.append(segment_path)

                # Update progress: 35% to 65% during segment extraction
                if progress_callback:
                    segment_progress = 35 + (idx / total_segments) * 30
                    progress_callback(int(segment_progress), 100)

            if not segment_files:
                raise ValueError("No segments were successfully extracted")

            # Update progress: 65% during concatenation
            if progress_callback:
                progress_callback(65, 100)

            output_path = self._combine_segments_with_crossfade(
                segment_files,
                output_path,
                output_format,
                quality,
                transition_duration=0.4,
            )

            if progress_callback:
                progress_callback(70, 100)

            corrected_output = self._finalize_output(
                output_path, float(target_duration), output_format, quality
            )

            logger.info(f"Quality time-lapse created: {corrected_output}")
            return corrected_output

        finally:
            # Clean up temporary files
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def add_music(
        self, video_path: str, music_path: str, output_path: str, volume: float = 0.5
    ):
        """Add music to video"""
        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(music_path).filter("volume", volume)

        output = ffmpeg.output(
            video.video, audio, output_path, vcodec="copy", acodec="aac"
        )
        output.run(overwrite_output=True)

        return output_path
