"""
Video Processor Module - FINAL STABLE VERSION
Correct duration + minimal dead frames + reliable NVENC.
"""

import cv2
import subprocess
import tempfile
import shutil
import os
import logging
import time
from typing import List, Tuple, Callable, Optional, Dict
from pathlib import Path
import platform
import numpy as np

logger = logging.getLogger(__name__)

QUALITY_PRESETS = [
    {"crf": "28", "nvenc_cq": "28", "preset": "veryfast"},
    {"crf": "23", "nvenc_cq": "23", "preset": "medium"},
    {"crf": "18", "nvenc_cq": "18", "preset": "slow"},
    {"crf": "15", "nvenc_cq": "15", "preset": "veryslow"},
]


class VideoProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.target_lengths: List[int] = config.get("target_lengths", [60])
        self.quality: int = config.get("quality", 2)
        self.use_gpu: bool = config.get("use_gpu", True)
        self.cpu_threads: int = config.get("cpu_threads", max(1, (os.cpu_count() or 4) - 1))
        self.add_music: bool = config.get("add_music", False)
        self.music_paths: dict = config.get("music_paths", {})
        self.motion_blur: bool = config.get("motion_blur", True)
        self.blur_strength: str = config.get("blur_strength", "light")

        self.cancelled: bool = False

        self.hw_accel = self._detect_hw_accel()
        logger.info(f"VideoProcessor ready | hw_accel={self.hw_accel} | quality={self.quality} | "
                   f"blur={self.blur_strength} | motion_blur={self.motion_blur}")

    def cancel(self):
        self.cancelled = True

    def create_timelapse_batch(self, input_path, motion_segments, output_base_path,
                              progress_callback=None, time_estimate_callback=None):
        results = {}
        t0 = time.time()
        base = Path(output_base_path)

        for i, target_length in enumerate(self.target_lengths):
            if self.cancelled:
                break
            suffix = {60: "_60s_vertical", 600: "_10min", 3600: "_1hour"}.get(
                target_length, f"_{target_length}s"
            )
            out_path = base.parent / f"{base.stem}{suffix}.mp4"

            def _prog(p):
                if progress_callback:
                    total = ((i + p / 100) / len(self.target_lengths)) * 100
                    progress_callback(total)

            ok = self.create_timelapse(
                input_path, motion_segments, str(out_path), target_length,
                _prog, time_estimate_callback
            )
            if ok:
                results[target_length] = str(out_path)

        return results

    def create_timelapse(self, input_path, motion_segments, output_path, target_length=None,
                        progress_callback=None, time_estimate_callback=None):
        try:
            if target_length is None:
                target_length = self.target_lengths[0]

            total_motion = sum(e - s for s, e in motion_segments)
            if total_motion == 0:
                logger.error("No motion segments")
                return False

            speedup = max(1.0, total_motion / target_length)
            logger.info(f"target={target_length}s | motion={total_motion:.1f}s | speedup={speedup:.2f}x")

            temp_dir = tempfile.mkdtemp()
            try:
                return self._process(input_path, motion_segments, output_path,
                                   target_length, speedup, temp_dir,
                                   progress_callback, time_estimate_callback)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            logger.exception("Error in create_timelapse")
            return False

    def _process(self, input_path, motion_segments, output_path,
                 target_length, speedup, temp_dir,
                 progress_callback, time_estimate_callback):
        t0 = time.time()
        segment_files = []

        for i, (start, end) in enumerate(motion_segments):
            if self.cancelled:
                return False
            if progress_callback:
                progress_callback((i / len(motion_segments)) * 50)
            if time_estimate_callback and i > 0:
                elapsed = time.time() - t0
                remaining = (elapsed / i) * (len(motion_segments) - i)
                time_estimate_callback(remaining)

            seg_path = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
            self._extract_segment(input_path, start, end, seg_path, speedup)
            segment_files.append(seg_path)

        if self.cancelled:
            return False
        if progress_callback:
            progress_callback(55)

        concat_path = os.path.join(temp_dir, "concat.mp4")
        self._concat(segment_files, concat_path)

        working = concat_path

        if self.add_music:
            music_path = self.music_paths.get(target_length)
            if music_path and os.path.exists(music_path):
                if progress_callback:
                    progress_callback(75)
                music_out = os.path.join(temp_dir, "with_music.mp4")
                self._add_music(working, music_path, music_out)
                working = music_out

        if target_length == 60:
            if progress_callback:
                progress_callback(85)
            rotated = os.path.join(temp_dir, "rotated.mp4")
            self._rotate(working, rotated)
            working = rotated

        if self.cancelled:
            return False
        if progress_callback:
            progress_callback(92)

        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"

        self._encode_final(working, output_path)
        logger.info(f"✅ Time-lapse created: {output_path}")
        return True

    def _extract_segment(self, input_path: str, start: float, end: float, output_path: str, speedup: float):
        duration = end - start
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_path, "-t", str(duration)]

        filters = ["setpts=PTS-STARTPTS"]

        if speedup > 1.0:
            filters.append(f"setpts={1.0/speedup:.6f}*PTS")

        if self.motion_blur and speedup > 1.5:
            if self.blur_strength == "strong":
                frames = max(2, min(8, int(speedup / 2)))
                weights = " ".join(["1.0"] * frames)
                filters.append(f"tmix=frames={frames}:weights='{weights}'")
            else:
                filters.append("tmix=frames=3:weights='1 1 1'")

        if filters:
            cmd += ["-vf", ",".join(filters)]

        cmd += self._encoder_args(fast=True) + ["-an", output_path]
        self._run(cmd)

    def _concat(self, segment_files: List[str], output_path: str):
        list_file = output_path.replace(".mp4", "_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for s in segment_files:
                f.write(f"file '{s}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path]
        self._run(cmd)
        try:
            os.remove(list_file)
        except OSError:
            pass

    def _add_music(self, video_path: str, music_path: str, output_path: str):
        vid_dur = self._probe_duration(video_path)
        mus_dur = self._probe_duration(music_path)
        if vid_dur <= 0 or mus_dur <= 0:
            logger.warning("Could not probe durations — skipping music")
            shutil.copy2(video_path, output_path)
            return

        loops = int(np.ceil(vid_dur / mus_dur))
        fade_out_start = max(0, vid_dur - 2)
        audio_filter = f"aloop=loop={loops}:size={int(mus_dur * 48000)},atrim=0:{vid_dur},afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out_start}:d=2"
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", music_path,
               "-filter_complex", audio_filter,
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path]
        self._run(cmd)

    def _rotate(self, input_path: str, output_path: str):
        cmd = ["ffmpeg", "-y", "-i", input_path,
               "-vf", "transpose=1", "-c:a", "copy", output_path]
        self._run(cmd)

    def _encode_final(self, input_path: str, output_path: str):
        """Final encode - reset timestamps to force exact target length"""
        preset = QUALITY_PRESETS[self.quality]
        cmd = ["ffmpeg", "-y", "-i", input_path,
               "-vf", "setpts=PTS-STARTPTS",   # ← This forces correct duration
               "-c:v", "h264_nvenc", "-preset", "p4",
               "-cq", preset["nvenc_cq"], "-b:v", "0", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-movflags", "+faststart", output_path]
        self._run(cmd)

    def _encoder_args(self, fast: bool = False, preset: dict = None) -> List[str]:
        if preset is None:
            preset = QUALITY_PRESETS[1]

        if self.hw_accel == "cuda":
            if fast:
                return ["-c:v", "h264_nvenc", "-preset", "p2", "-pix_fmt", "yuv420p"]
            return ["-c:v", "h264_nvenc", "-preset", "p4",
                    "-cq", preset["nvenc_cq"], "-b:v", "0", "-pix_fmt", "yuv420p"]

        return ["-c:v", "libx264",
                "-preset", "ultrafast" if fast else preset["preset"],
                "-crf", preset["crf"],
                "-threads", str(self.cpu_threads),
                "-pix_fmt", "yuv420p"]

    def _detect_hw_accel(self) -> str:
        if not self.use_gpu:
            return "cpu"
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
            if r.returncode == 0:
                enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                     capture_output=True, text=True, timeout=3)
                if "h264_nvenc" in enc.stdout:
                    logger.info("✅ NVENC (CUDA) encoder detected")
                    return "cuda"
        except Exception:
            pass
        logger.info("Using CPU encoding")
        return "cpu"

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=10)
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def _run(cmd: List[str]):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed (code {e.returncode}): {' '.join(cmd)}")
            stderr = e.stderr.decode(errors="replace")
            logger.error(f"stderr: {stderr[-2000:]}")
            raise