"""
Bird Timelapse Video Processor  v16.0
═══════════════════════════════════════════════════════════════════════════════

PIPELINE STAGES
───────────────
Stage 1  Extract segments  — fast stream-copy per segment, no re-encoding.
Stage 2  Concatenate       — stitch segment files, reset timestamps.
Stage 3  Post-process      — deflicker + eq + denoise + motion blur.
Stage 4  Final encode      — speedup + rotate (Shorts) + watermark +
                             music mix + libx264/NVENC, hard -t cap.

Stages 3 and 4 are ONE FFmpeg command on the concatenated clip.
There is never a second re-encode.

WHY THIS ORDER
──────────────
• Motion detection first (in MotionDetector) → only interesting frames pass.
• Stream-copy extraction → lossless, fast.
• All heavy filters applied once on the motion-only clip → minimum processing.
• Everything from deflicker to final encode in a single FFmpeg pass → best
  quality (no generation loss), fastest overall wall-clock time.
• tmix (blur) is applied BEFORE setpts (speedup) — see filters.py for why.
• Rotation is the last geometric operation so earlier filters process fewer
  pixels in total.
"""

import os
import subprocess
import tempfile
import shutil
import logging
import threading
import time
from typing import List, Tuple, Optional, Union

from filters import (
    build_vf,
    build_af,
    build_filter_complex_with_music,
    recommended_blur_frames,
)

logger = logging.getLogger(__name__)

QUALITY_MAP = {
    "Low":      {"crf": "28", "preset": "veryfast", "nv_preset": "p2"},
    "Medium":   {"crf": "23", "preset": "medium",   "nv_preset": "p4"},
    "High":     {"crf": "18", "preset": "slow",     "nv_preset": "p5"},
    "Maximum":  {"crf": "15", "preset": "veryslow", "nv_preset": "p6"},
    0: {"crf": "28", "preset": "veryfast", "nv_preset": "p2"},
    1: {"crf": "23", "preset": "medium",   "nv_preset": "p4"},
    2: {"crf": "18", "preset": "slow",     "nv_preset": "p5"},
    3: {"crf": "15", "preset": "veryslow", "nv_preset": "p6"},
}


class VideoProcessor:
    """
    Four-stage bird timelapse pipeline.

    Usage
    ─────
    proc = VideoProcessor(config)
    proc.set_cancel_flag(threading.Event())
    ok   = proc.create_timelapse(
               input_path, segments, output_path,
               target_length=59,
               music_path="/path/to/track.mp3",
               progress_callback=cb,
           )
    """

    def __init__(self, config: dict):
        self.config      = config
        self.ff          = self._find_ffmpeg()
        self.nvenc_ok    = self._check_nvenc()
        self._cancel:    Optional[threading.Event] = None
        self._proc:      Optional[subprocess.Popen] = None
        self._proc_lock  = threading.Lock()

        logger.info(
            f"VideoProcessor v16.0 | "
            f"nvenc={self.nvenc_ok} "
            f"quality={config.get('quality', 'High')} "
            f"gpu={config.get('use_gpu', True)}"
        )

    def set_cancel_flag(self, f: threading.Event) -> None:
        self._cancel = f

    def cancel(self) -> None:
        """Immediately kill the running FFmpeg subprocess."""
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    logger.info("[PROC] FFmpeg subprocess terminated by cancel()")
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────── public API ──

    def create_timelapse(
        self,
        input_path:         str,
        segments:           List[Tuple[float, float]],
        output_path:        str,
        target_length:      float = 59.0,
        music_path:         Optional[Union[str, List[str]]] = None,
        progress_callback:  Optional[callable] = None,
        status_callback:    Optional[callable] = None,
    ) -> bool:
        """
        Run the full pipeline.  Returns True on success.
        target_length ≤ 60 → video is rotated 90° CW (vertical Short).
        """
        # Normalise music
        if isinstance(music_path, list):
            music_path = music_path[0] if music_path else None
        music_path = str(music_path).strip() if music_path else None
        if music_path and not os.path.exists(music_path):
            logger.warning(f"[PROC] Music not found, skipping: {music_path}")
            music_path = None
        # Also check config music_paths
        if not music_path:
            mp = self.config.get("music_paths", {})
            for k in (int(target_length), str(int(target_length)), "default"):
                v = mp.get(k)
                if v and os.path.exists(str(v)):
                    music_path = str(v)
                    break

        if not segments:
            logger.error("[PROC] No segments provided")
            return False

        total_motion = sum(e - s for s, e in segments)
        if total_motion <= 0:
            logger.error("[PROC] Zero total motion duration")
            return False

        # 2 % safety margin; -t is the hard ceiling
        speed_factor = max(1.0, (total_motion / target_length) * 1.02)
        is_short     = target_length <= 60

        if not output_path.endswith(".mp4"):
            output_path += ".mp4"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        logger.info(
            f"[PROC] Pipeline start | "
            f"segments={len(segments)} "
            f"motion={total_motion:.1f}s "
            f"target={target_length}s "
            f"speed={speed_factor:.3f}x "
            f"is_short={is_short} "
            f"music={'yes' if music_path else 'no'}"
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
            logger.debug(f"[PROC] Temp dir cleaned: {tmp}")

    # ──────────────────────────────────────────────── pipeline stages ────

    def _pipeline(
        self, src, segs, dst, target_len, spd, is_short, music,
        tmp, pcb, scb,
    ) -> bool:
        t0 = time.time()

        def _p(v):
            if pcb: pcb(v)

        def _s(m):
            logger.info(f"[PROC] Status: {m}")
            if scb: scb(m)

        # ── Stage 1: Extract segments (stream-copy) ───────────────────────
        _s(f"Extracting {len(segs)} motion segments…")
        seg_files = self._extract_segments(src, segs, tmp,
                                           pcb=lambda v: _p(v * 0.30))
        if not seg_files:
            logger.error("[PROC] Stage 1 failed — no segments extracted")
            return False
        _p(30)
        logger.info(
            f"[PROC] Stage 1 done: "
            f"{len(seg_files)}/{len(segs)} segments extracted "
            f"({time.time()-t0:.1f}s)"
        )

        # ── Stage 2: Concatenate ─────────────────────────────────────────
        _s("Concatenating segments…")
        concat = os.path.join(tmp, "concat.mp4")
        if not self._concat(seg_files, concat):
            logger.error("[PROC] Stage 2 failed — concat error")
            return False
        _p(40)
        logger.info(f"[PROC] Stage 2 done: concat ({time.time()-t0:.1f}s)")

        # ── Stages 3+4: Post-process + Final encode (single FFmpeg pass) ──
        _s(
            f"Encoding {target_len}s "
            + (f"Short " if is_short else "")
            + f"at {spd:.1f}× speed…"
        )
        ok = self._encode(concat, dst, target_len, spd, is_short, music,
                          pcb=lambda v: _p(40 + v * 0.60))
        if ok:
            elapsed = time.time() - t0
            dur = self._probe_duration(dst)
            logger.info(
                f"[PROC] Pipeline complete | "
                f"output={os.path.basename(dst)} "
                f"duration={dur:.2f}s "
                f"target={target_len}s "
                f"total_elapsed={elapsed:.1f}s"
            )
        else:
            logger.error(f"[PROC] Stage 3/4 failed — encoding error")
        return ok

    # ── Stage 1 ───────────────────────────────────────────────────────────

    def _extract_segments(
        self, src, segs, tmp, pcb=None
    ) -> List[str]:
        files = []
        for i, (start, end) in enumerate(segs):
            if self._cancelled(): return []
            out = os.path.join(tmp, f"s{i:04d}.mp4")
            ok  = self._run([
                self.ff, "-y",
                "-ss", f"{start:.6f}",
                "-i",  src,
                "-t",  f"{end - start:.6f}",
                "-c",  "copy",
                "-avoid_negative_ts", "make_zero",
                out,
            ], timeout=120)
            if ok and os.path.exists(out) and os.path.getsize(out) > 2000:
                files.append(out)
                logger.debug(
                    f"[PROC] Segment {i}: "
                    f"{start:.2f}s–{end:.2f}s "
                    f"({end-start:.2f}s)"
                )
            else:
                logger.warning(f"[PROC] Segment {i} extraction failed or empty")
            if pcb: pcb((i + 1) / len(segs) * 100)
        logger.info(f"[PROC] Stage 1: extracted {len(files)}/{len(segs)} segments")
        return files

    # ── Stage 2 ───────────────────────────────────────────────────────────

    def _concat(self, files, out) -> bool:
        lst = out.replace(".mp4", "_list.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            for f in files:
                fh.write(f"file '{f}'\n")
        ok = self._run([
            self.ff, "-y",
            "-f", "concat", "-safe", "0",
            "-i", lst,
            "-c", "copy",
            "-reset_timestamps", "1",
            out,
        ], timeout=600)
        try:
            os.remove(lst)
        except OSError:
            pass
        return ok

    # ── Stages 3+4 ────────────────────────────────────────────────────────

    def _encode(
        self, inp, dst, target_len, spd, is_short, music, pcb=None
    ) -> bool:
        """
        Single FFmpeg command that covers:
          deflicker → eq → hqdn3d → tmix blur → setpts speedup →
          transpose (Short) → watermark → music mix → encode → -t cap
        """
        cfg = self.config
        q   = QUALITY_MAP.get(cfg.get("quality", "High"), QUALITY_MAP["High"])

        # Motion blur frames: auto if not overridden by user
        blur_user = cfg.get("motion_blur_frames", -1)
        blur_frames = (
            blur_user if (blur_user >= 0 and is_short)
            else (recommended_blur_frames(spd) if is_short else 0)
        )

        vf = build_vf(
            speed_factor   = spd,
            is_short       = is_short,
            blur_frames    = blur_frames,
            deflicker_size = int(cfg.get("deflicker_size", 5)),
            contrast       = float(cfg.get("contrast",    1.0)),
            saturation     = float(cfg.get("saturation",  1.0)),
            brightness     = float(cfg.get("brightness",  0.0)),
            denoise        = bool(cfg.get("denoise",       True)),
            watermark      = cfg.get("watermark_text") or None,
        )

        logger.info(
            f"[PROC] Encode params | "
            f"speed={spd:.3f}x "
            f"blur={blur_frames}fr "
            f"is_short={is_short} "
            f"deflicker={cfg.get('deflicker_size',5)} "
            f"contrast={cfg.get('contrast',1.0)} "
            f"sat={cfg.get('saturation',1.0)} "
            f"brightness={cfg.get('brightness',0.0)}"
        )

        # Encoder
        use_nvenc = self.nvenc_ok and cfg.get("use_gpu", True)
        if use_nvenc:
            enc = ["-c:v", "h264_nvenc", "-preset", q["nv_preset"],
                   "-cq", q["crf"], "-b:v", "0"]
        else:
            enc = ["-c:v", "libx264", "-preset", q["preset"],
                   "-crf", q["crf"],
                   "-threads", str(cfg.get("cpu_threads", 0) or 0)]
        enc += ["-pix_fmt", "yuv420p"]

        # Extra custom args
        custom = cfg.get("custom_ffmpeg_args", "")
        extra  = custom.split() if custom else []

        if music:
            fc = build_filter_complex_with_music(
                speed_factor   = spd,
                target_length  = target_len,
                music_volume   = float(cfg.get("music_volume", 0.5)),
                vf_chain       = vf,
            )
            cmd = [
                self.ff, "-y",
                "-i", inp,
                "-i", music,
                "-filter_complex", fc,
                "-map", "[vout]",
                "-map", "[aout]",
            ]
        else:
            af = build_af(spd, target_len)
            cmd = [
                self.ff, "-y",
                "-i", inp,
                "-vf", vf,
                "-af", af,
            ]

        cmd += enc + extra + [
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(target_len),       # ← hard cap: NEVER exceed target
            "-movflags", "+faststart",
            dst,
        ]

        logger.debug(f"[PROC] FFmpeg cmd: {' '.join(cmd)}")
        ok = self._run(cmd, timeout=7200)
        if pcb and ok: pcb(100)
        return ok

    # ──────────────────────────────────────────────────────── helpers ─────

    def _run(self, cmd: List[str], timeout: int = 300) -> bool:
        if self._cancelled(): return False
        try:
            with self._proc_lock:
                p = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                self._proc = p
            _, err = p.communicate(timeout=timeout)
            with self._proc_lock:
                self._proc = None
            if p.returncode != 0:
                if self._cancelled(): return False
                logger.error(
                    f"[PROC] FFmpeg error (rc={p.returncode}): "
                    f"{' '.join(cmd[:5])}…\n"
                    + err.decode(errors="replace")[-2000:]
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            with self._proc_lock:
                if self._proc:
                    self._proc.terminate()
                    self._proc = None
            logger.error(f"[PROC] FFmpeg timeout after {timeout}s: {' '.join(cmd[:5])}")
            return False
        except Exception as exc:
            logger.exception(f"[PROC] FFmpeg exception: {exc}")
            return False

    def _cancelled(self) -> bool:
        return bool(self._cancel and self._cancel.is_set())

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=15,
            )
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def _find_ffmpeg() -> str:
        for p in ["ffmpeg",
                  os.path.join("ffmpeg", "bin", "ffmpeg.exe"),
                  r"C:\ffmpeg\bin\ffmpeg.exe"]:
            try:
                subprocess.run([p, "-version"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=3)
                return p
            except Exception:
                continue
        return "ffmpeg"

    def _check_nvenc(self) -> bool:
        try:
            r = subprocess.run(
                [self.ff, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5,
            )
            ok = "h264_nvenc" in r.stdout
            logger.info(f"[PROC] NVENC available: {ok}")
            return ok
        except Exception:
            return False

    # backward-compat alias
    def process_single_video(self, *a, **kw):
        return self.create_timelapse(*a, **kw)
