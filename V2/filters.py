"""
filters.py — FFmpeg filter string builders
══════════════════════════════════════════════════════════════════════════════

All VF and AF filter strings are constructed here so that video_processor.py
stays clean and the filter logic is unit-testable in isolation.

CORRECT ORDER OF OPERATIONS inside the VF chain
────────────────────────────────────────────────
1. setpts=PTS-STARTPTS          — reset PTS drift from concat
2. deflicker                    — remove per-frame brightness spikes
3. eq (contrast/sat/brightness) — global colour grade
4. hqdn3d                       — spatial+temporal denoise
5. tmix (motion blur)           — temporal frame blending  ← BEFORE speedup
6. setpts=PTS/SPEED             — speedup  ← AFTER blur
7. transpose=1                  — rotate for Shorts  ← after speedup
8. drawtext (watermark)         — overlay on final geometry

Why this order?
  • deflicker/eq/hqdn3d work on real-time content → must come before speedup.
  • tmix blends N source frames into one output frame, simulating a longer
    shutter.  If applied AFTER setpts the frames are already "virtual" and
    the blur is meaningless.
  • Rotation is a geometric transform; doing it last avoids processing extra
    pixels in all previous filters.
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────── Video filter builders ────────────────

def build_vf(
    speed_factor:      float,
    is_short:          bool  = False,
    blur_frames:       int   = 0,
    deflicker_size:    int   = 5,
    contrast:          float = 1.0,
    saturation:        float = 1.0,
    brightness:        float = 0.0,
    denoise:           bool  = True,
    watermark:         Optional[str] = None,
) -> str:
    """
    Build the complete -vf filter chain.

    Parameters
    ──────────
    speed_factor    PTS divisor for speedup.  Must be ≥ 1.0.
    is_short        True → append transpose=1 (90° CW for vertical Shorts).
    blur_frames     Number of frames for tmix motion blur (0 = disabled).
    deflicker_size  deflicker sliding window (5 is a good default).
    contrast        eq contrast (1.0 = neutral, 1.05–1.2 = slightly punchy).
    saturation      eq saturation (1.0 = neutral, 1.05–1.15 = warmer colours).
    brightness      eq brightness offset (-1.0 – +1.0, 0 = neutral).
    denoise         Apply hqdn3d light denoising.
    watermark       Watermark text string or None.

    Returns a comma-separated FFmpeg filter string ready for -vf "...".
    """
    parts = []

    # 1. Reset PTS drift from concat / deflicker encode
    parts.append("setpts=PTS-STARTPTS")

    # 2. Deflicker
    if deflicker_size > 0:
        parts.append(f"deflicker=size={deflicker_size}:mode=am")

    # 3. eq (colour grade) — only emit if any value is non-neutral
    if contrast != 1.0 or saturation != 1.0 or brightness != 0.0:
        parts.append(
            f"eq=contrast={contrast:.3f}"
            f":saturation={saturation:.3f}"
            f":brightness={brightness:.3f}"
        )

    # 4. Denoise
    if denoise:
        parts.append("hqdn3d=luma_spatial=1.5:luma_tmp=2.5:chroma_spatial=1:chroma_tmp=2")

    # 5. Motion blur (tmix) — BEFORE speedup
    if blur_frames >= 2:
        weights = _gaussian_weights(blur_frames)
        parts.append(f"tmix=frames={blur_frames}:weights='{weights}'")
        logger.debug(f"[FILTER] tmix frames={blur_frames} weights={weights}")

    # 6. Speedup — AFTER blur
    if speed_factor > 1.001:
        parts.append(f"setpts=PTS/{speed_factor:.6f}")

    # 7. Rotate for vertical Short
    if is_short:
        parts.append("transpose=1")

    # 8. Watermark
    if watermark:
        safe = watermark.replace("'", r"\'").replace(":", r"\:")
        parts.append(
            f"drawtext=text='{safe}'"
            f":fontcolor=white@0.8:fontsize=22"
            f":x=10:y=10"
            f":shadowcolor=black@0.6:shadowx=1:shadowy=1"
        )

    vf = ",".join(parts)
    logger.debug(f"[FILTER] vf chain: {vf}")
    return vf


# ─────────────────────────────────────── Audio filter builders ───────────────

def build_af(
    speed_factor: float,
    target_length: float,
    fade_out_sec: float = 2.0,
) -> str:
    """
    Build the -af chain for video-only audio (no music):
      atempo chain → afade out at end.
    """
    parts = [_atempo_chain(speed_factor)]
    fade_start = max(0.0, target_length - fade_out_sec)
    parts.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.1f}")
    af = ",".join(parts)
    logger.debug(f"[FILTER] af chain: {af}")
    return af


def build_filter_complex_with_music(
    speed_factor:  float,
    target_length: float,
    music_volume:  float = 0.5,
    fade_out_sec:  float = 2.0,
    vf_chain:      str   = "",
) -> str:
    """
    Build -filter_complex when mixing original audio with a music track.

    Input streams assumed:
      [0:v] [0:a]  — concat video/audio
      [1:a]        — music track

    Returns the full filter_complex string.
    """
    # Video: full vf chain on stream 0
    fc_parts = [f"[0:v]{vf_chain}[vout]"]

    # Original audio: speed up
    af_orig = _atempo_chain(speed_factor)
    fade_start = max(0.0, target_length - fade_out_sec)
    fc_parts.append(
        f"[0:a]{af_orig},"
        f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.1f}[aorig]"
    )

    # Music: loop infinitely, trim to target, volume, fade
    fc_parts.append(
        f"[1:a]aloop=loop=-1:size=2e+09,"
        f"atrim=0:{target_length:.3f},"
        f"volume={music_volume:.3f},"
        f"afade=t=in:st=0:d=0.5,"
        f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.1f}[amusic]"
    )

    # Mix
    fc_parts.append(
        "[aorig][amusic]amix=inputs=2:duration=first"
        ":dropout_transition=2:weights='1 1'[aout]"
    )

    fc = ";".join(fc_parts)
    logger.debug(f"[FILTER] filter_complex: {fc}")
    return fc


# ─────────────────────────────────────── Motion blur helpers ─────────────────

def recommended_blur_frames(speed_factor: float) -> int:
    """
    Return the recommended tmix frame count for a given speed factor.

    Rationale: at Nx speedup each output frame represents N source frames.
    Blending those N frames approximates a longer physical shutter.
    We cap at 8 because beyond that the result is unconvincing smear.

    speed < 2.0  →  0  (barely sped up; natural motion is fine)
    2.0 – 3.0   →  3
    3.0 – 5.0   →  4
    5.0 – 8.0   →  6
    > 8.0        →  8  (max)
    """
    if speed_factor < 2.0:
        return 0
    if speed_factor < 3.0:
        return 3
    if speed_factor < 5.0:
        return 4
    if speed_factor < 8.0:
        return 6
    return 8


def _gaussian_weights(n: int) -> str:
    """
    Gaussian weight vector for tmix.

    Equal weights ("1 1 1") cause the background to ghost through at full
    opacity.  A Gaussian centred on the middle frame keeps the main subject
    sharp while the motion trail fades naturally — a much more cinematic look.
    """
    if n <= 1:
        return "1"
    sigma   = (n - 1) / 4.0
    weights = [
        math.exp(-0.5 * ((i - (n - 1) / 2) / sigma) ** 2)
        for i in range(n)
    ]
    total = sum(weights)
    norm  = [w / total * n for w in weights]
    return " ".join(f"{w:.4f}" for w in norm)


def _atempo_chain(speed: float) -> str:
    """
    Build a chained atempo filter for any speed ≥ 0.5.
    Each atempo stage is limited to [0.5, 2.0] by FFmpeg.

    Examples:
      1.5  → "atempo=1.500000"
      3.0  → "atempo=2.000000,atempo=1.500000"
      8.0  → "atempo=2.000000,atempo=2.000000,atempo=2.000000"
    """
    stages: list = []
    rem = float(speed)
    while rem > 2.0 + 1e-6:
        stages.append("atempo=2.000000")
        rem /= 2.0
    while rem < 0.5 - 1e-6:
        stages.append("atempo=0.500000")
        rem /= 0.5
    stages.append(f"atempo={rem:.6f}")
    return ",".join(stages)
