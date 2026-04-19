"""
Bird Timelapse Video Processor  v17.0  (VideoProcessor / VideoPipeline)
═══════════════════════════════════════════════════════════════════════════════

PIPELINE STAGES
───────────────
Stage 1  Extract segments  — fast stream-copy, no re-encoding.
Stage 2  Concatenate       — stitch, reset timestamps.
Stage 3+4  Post-process + Final encode  — ONE FFmpeg command:
           deflicker → eq → hqdn3d → tmix blur → setpts speedup →
           film_grain → vignette → lut/grade → rotate (Shorts) →
           watermark → music mix → libx264/NVENC → hard -t cap.
"""

import os
import subprocess
import tempfile
import shutil
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from filters import (
    build_vf,
    build_af,
    build_filter_complex_with_music,
    recommended_blur_frames,
)

logger = logging.getLogger(__name__)

QUALITY_MAP = {
    "Low":     {"crf": "28", "preset": "veryfast", "nv_preset": "p2"},
    "Medium":  {"crf": "23", "preset": "medium",   "nv_preset": "p4"},
    "High":    {"crf": "18", "preset": "slow",     "nv_preset": "p5"},
    "Maximum": {"crf": "15", "preset": "veryslow", "nv_preset": "p6"},
    0: {"crf": "28", "preset": "veryfast", "nv_preset": "p2"},
    1: {"crf": "23", "preset": "medium",   "nv_preset": "p4"},
    2: {"crf": "18", "preset": "slow",     "nv_preset": "p5"},
    3: {"crf": "15", "preset": "veryslow", "nv_preset": "p6"},
}


class VideoProcessor:
    """Four-stage bird timelapse pipeline v17.0. Alias: VideoPipeline."""

    def __init__(self, config: dict):
        self.config     = config
        self.ff         = self._find_ffmpeg()
        self.nvenc_ok   = self._check_nvenc()
        self._cancel:   Optional[threading.Event] = None
        self._proc:     Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        logger.info(
            f"VideoProcessor v17.0 | nvenc={self.nvenc_ok} "
            f"quality={config.get('quality','High')} gpu={config.get('use_gpu',True)}"
        )

    def set_cancel_flag(self, f: threading.Event) -> None:
        self._cancel = f

    def cancel(self) -> None:
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    logger.info("[PROC] Terminated by cancel()")
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────── public API ──

    def create_timelapse(
        self,
        input_path:        str,
        segments:          List[Tuple[float, float]],
        output_path:       str,
        target_length:     float = 59.0,
        music_path:        Optional[Union[str, List[str]]] = None,
        progress_callback: Optional[callable] = None,
        status_callback:   Optional[callable] = None,
    ) -> bool:
        if isinstance(music_path, list):
            music_path = music_path[0] if music_path else None
        music_path = str(music_path).strip() if music_path else None
        if music_path and not os.path.exists(music_path):
            logger.warning(f"[PROC] Music not found: {music_path}")
            music_path = None
        if not music_path:
            mp = self.config.get("music_paths", {})
            for k in (int(target_length), str(int(target_length)), "default"):
                v = mp.get(k)
                if v and os.path.exists(str(v)):
                    music_path = str(v); break

        if not segments:
            logger.error("[PROC] No segments"); return False
        total_motion = sum(e - s for s, e in segments)
        if total_motion <= 0:
            logger.error("[PROC] Zero motion"); return False

        speed_factor = max(1.0, (total_motion / target_length) * 1.02)
        is_short     = target_length <= 60
        if not output_path.endswith(".mp4"):
            output_path += ".mp4"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        logger.info(
            f"[PROC] Start | segs={len(segments)} motion={total_motion:.1f}s "
            f"target={target_length}s speed={speed_factor:.3f}x is_short={is_short}"
        )
        tmp = tempfile.mkdtemp(prefix="bbvp_")
        try:
            return self._pipeline(
                input_path, segments, output_path,
                target_length, speed_factor, is_short, music_path,
                tmp, progress_callback, status_callback,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def validate_input(self, path: str) -> dict:
        try:
            import json as _json
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate,codec_name:format=duration",
                 "-of", "json", path],
                capture_output=True, text=True, timeout=15,
            )
            data   = _json.loads(r.stdout)
            stream = data.get("streams", [{}])[0]
            fmt    = data.get("format", {})
            fps_n, fps_d = stream.get("r_frame_rate", "30/1").split("/")
            return {
                "fps":      float(fps_n) / max(float(fps_d), 1),
                "duration": float(fmt.get("duration", 0)),
                "width":    int(stream.get("width", 0)),
                "height":   int(stream.get("height", 0)),
                "codec":    stream.get("codec_name", "unknown"),
            }
        except Exception as exc:
            logger.warning(f"[PROC] validate_input: {exc}")
            return {}

    def build_final_command(
        self,
        processed_path: str,
        output_path:    str,
        target_length:  float,
        music_path:     Optional[str],
        speed_factor:   float,
        is_short:       bool = False,
    ) -> List[str]:
        cfg         = self.config
        q           = QUALITY_MAP.get(cfg.get("quality", "High"), QUALITY_MAP["High"])
        blur_user   = cfg.get("motion_blur_frames", -1)
        blur_frames = (
            blur_user if (blur_user >= 0 and is_short)
            else (recommended_blur_frames(speed_factor) if is_short else 0)
        )
        vf = build_vf(
            speed_factor       = speed_factor,
            is_short           = is_short,
            blur_frames        = blur_frames,
            deflicker_size     = int(cfg.get("deflicker_size", 5)),
            contrast           = float(cfg.get("contrast", 1.0)),
            saturation         = float(cfg.get("saturation", 1.0)),
            brightness         = float(cfg.get("brightness", 0.0)),
            denoise            = bool(cfg.get("denoise", True)),
            watermark          = cfg.get("watermark_text") or None,
            film_grain         = float(cfg.get("film_grain", 0.0)),
            vignette_strength  = float(cfg.get("vignette_strength", 0.0)),
            lut_path           = cfg.get("lut_path") or None,
            color_grade        = cfg.get("color_grade_preset") or None,
        )
        use_nvenc = self.nvenc_ok and cfg.get("use_gpu", True)
        enc = (
            ["-c:v", "h264_nvenc", "-preset", q["nv_preset"], "-cq", q["crf"], "-b:v", "0"]
            if use_nvenc else
            ["-c:v", "libx264", "-preset", q["preset"], "-crf", q["crf"],
             "-threads", str(cfg.get("cpu_threads", 0) or 0)]
        )
        enc += ["-pix_fmt", "yuv420p"]

        include_orig = bool(cfg.get("include_original_audio", False))
        if music_path:
            fc  = build_filter_complex_with_music(
                speed_factor     = speed_factor,
                target_length    = target_length,
                music_volume     = float(cfg.get("music_volume", 0.5)),
                vf_chain         = vf,
                include_original = include_orig,
            )
            cmd = [self.ff, "-y", "-i", processed_path, "-i", music_path,
                   "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]"]
        else:
            af  = build_af(speed_factor, target_length, include_original=include_orig)
            cmd = [self.ff, "-y", "-i", processed_path, "-vf", vf, "-af", af]

        extra = (cfg.get("custom_ffmpeg_args") or "").split()
        cmd  += enc + extra + [
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(target_length),
            "-movflags", "+faststart",
            output_path,
        ]
        return cmd

    def run_with_progress(self, cmd, progress_callback=None, cancel_flag=None, timeout=7200) -> bool:
        return self._run(cmd, timeout=timeout)

    # ──────────────────────────────────────────────────────── pipeline ────

    def _pipeline(self, src, segs, dst, target_len, spd, is_short, music, tmp, pcb, scb) -> bool:
        t0 = time.time()

        def _p(v):
            if pcb: pcb(v)

        def _s(m):
            logger.info(f"[PROC] {m}")
            if scb: scb(m)

        _s(f"Extracting {len(segs)} segments…")
        seg_files = self._extract_segments(src, segs, tmp, pcb=lambda v: _p(v * 0.30))
        if not seg_files:
            logger.error("[PROC] Stage 1 failed"); return False
        _p(30)

        _s("Concatenating…")
        concat = os.path.join(tmp, "concat.mp4")
        if not self._concat(seg_files, concat):
            logger.error("[PROC] Stage 2 failed"); return False
        _p(40)

        _s(f"Encoding {target_len}s {'Short ' if is_short else ''}at {spd:.1f}×…")
        cmd = self.build_final_command(concat, dst, target_len, music, spd, is_short)
        logger.debug(f"[PROC] cmd: {' '.join(cmd)}")
        ok  = self._run(cmd, timeout=7200)
        if ok:
            _p(100)
            dur = self._probe_duration(dst)
            logger.info(
                f"[PROC] Complete | {Path(dst).name} "
                f"duration={dur:.2f}s elapsed={time.time()-t0:.1f}s"
            )
        else:
            logger.error("[PROC] Encode failed")
        return ok

    def _extract_segments(self, src, segs, tmp, pcb=None) -> List[str]:
        files = []
        for i, (start, end) in enumerate(segs):
            if self._cancelled(): return []
            out = os.path.join(tmp, f"s{i:04d}.mp4")
            ok  = self._run([
                self.ff, "-y",
                "-ss", f"{start:.6f}", "-i", src,
                "-t",  f"{end - start:.6f}",
                "-c",  "copy", "-avoid_negative_ts", "make_zero", out,
            ], timeout=120)
            if ok and os.path.exists(out) and os.path.getsize(out) > 2000:
                files.append(out)
            else:
                logger.warning(f"[PROC] Segment {i} failed/empty")
            if pcb: pcb((i + 1) / len(segs) * 100)
        return files

    def _concat(self, files, out) -> bool:
        lst = out.replace(".mp4", "_list.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            for f in files:
                fh.write(f"file '{f}'\n")
        ok = self._run([
            self.ff, "-y", "-f", "concat", "-safe", "0", "-i", lst,
            "-c", "copy", "-reset_timestamps", "1", out,
        ], timeout=600)
        try: os.remove(lst)
        except OSError: pass
        return ok

    def _run(self, cmd: List[str], timeout: int = 300) -> bool:
        if self._cancelled(): return False
        try:
            with self._proc_lock:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._proc = p
            _, err = p.communicate(timeout=timeout)
            with self._proc_lock:
                self._proc = None
            if p.returncode != 0:
                if self._cancelled(): return False
                logger.error(
                    f"[PROC] FFmpeg rc={p.returncode}: "
                    + err.decode(errors="replace")[-2000:]
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            with self._proc_lock:
                if self._proc: self._proc.terminate(); self._proc = None
            logger.error(f"[PROC] Timeout after {timeout}s")
            return False
        except Exception as exc:
            logger.exception(f"[PROC] Exception: {exc}")
            return False

    def _cancelled(self) -> bool:
        return bool(self._cancel and self._cancel.is_set())

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=15,
            )
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def _find_ffmpeg() -> str:
        for p in ["ffmpeg", os.path.join("ffmpeg","bin","ffmpeg.exe"), r"C:\ffmpeg\bin\ffmpeg.exe"]:
            try:
                subprocess.run([p, "-version"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=3)
                return p
            except Exception:
                continue
        return "ffmpeg"

    def _check_nvenc(self) -> bool:
        try:
            r = subprocess.run([self.ff, "-hide_banner", "-encoders"],
                               capture_output=True, text=True, timeout=5)
            ok = "h264_nvenc" in r.stdout
            logger.info(f"[PROC] NVENC: {ok}")
            return ok
        except Exception:
            return False

    def process_single_video(self, *a, **kw):
        return self.create_timelapse(*a, **kw)


VideoPipeline = VideoProcessor
