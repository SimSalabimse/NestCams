"""
filters.py — FFmpeg filter string builders  v2.0
══════════════════════════════════════════════════════════════════════════════

CORRECT VF ORDER (must never change):
  1. setpts=PTS-STARTPTS   — reset PTS drift
  2. deflicker             — per-frame brightness spikes
  3. eq                    — colour grade
  4. hqdn3d                — denoise
  5. tmix                  — motion blur  ← BEFORE speedup
  6. setpts=PTS/SPEED      — speedup      ← AFTER blur
  7. noise                 — film grain
  8. vignette              — edge darkening
  9. lut3d / eq grade      — LUT or built-in preset
 10. transpose=1           — rotate for Shorts
 11. drawtext              — watermark
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
    film_grain:        float = 0.0,
    vignette_strength: float = 0.0,
    lut_path:          Optional[str] = None,
    color_grade:       Optional[str] = None,
) -> str:
    parts = []

    parts.append("setpts=PTS-STARTPTS")

    if deflicker_size > 0:
        parts.append(f"deflicker=size={deflicker_size}:mode=am")

    if contrast != 1.0 or saturation != 1.0 or brightness != 0.0:
        parts.append(
            f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}:brightness={brightness:.3f}"
        )

    if denoise:
        parts.append("hqdn3d=luma_spatial=1.5:luma_tmp=2.5:chroma_spatial=1:chroma_tmp=2")

    if blur_frames >= 2:
        weights = _gaussian_weights(blur_frames)
        parts.append(f"tmix=frames={blur_frames}:weights='{weights}'")
        logger.debug(f"[FILTER] tmix frames={blur_frames}")

    if speed_factor > 1.001:
        parts.append(f"setpts=PTS/{speed_factor:.6f}")

    if film_grain > 0.001:
        parts.append(f"noise=alls={int(film_grain * 100)}:allf=t+u")

    if vignette_strength > 0.01:
        angle = math.pi / (2.0 + (0.6 - min(0.6, vignette_strength)) * 3)
        parts.append(f"vignette=angle={angle:.4f}:mode=forward")

    if lut_path:
        safe = lut_path.replace("\\", "/").replace("'", r"\'")
        parts.append(f"lut3d=file='{safe}'")
    elif color_grade and color_grade != "None":
        grade_filter = _builtin_grade(color_grade)
        if grade_filter:
            parts.append(grade_filter)

    if is_short:
        parts.append("transpose=1")

    if watermark:
        safe = watermark.replace("'", r"\'").replace(":", r"\:")
        parts.append(
            f"drawtext=text='{safe}':fontcolor=white@0.8:fontsize=22"
            ":x=10:y=10:shadowcolor=black@0.6:shadowx=1:shadowy=1"
        )

    vf = ",".join(parts)
    logger.debug(f"[FILTER] vf chain ({len(parts)} filters): {vf[:120]}")
    return vf


def build_af(
    speed_factor:     float,
    target_length:    float,
    fade_out_sec:     float = 2.0,
    include_original: bool  = False,
) -> str:
    parts = [_atempo_chain(speed_factor)]
    if include_original:
        parts.append("volume=0.3")
    fade_start = max(0.0, target_length - fade_out_sec)
    parts.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.1f}")
    af = ",".join(parts)
    logger.debug(f"[FILTER] af chain: {af}")
    return af


def build_filter_complex_with_music(
    speed_factor:     float,
    target_length:    float,
    music_volume:     float = 0.5,
    fade_out_sec:     float = 2.0,
    vf_chain:         str   = "",
    include_original: bool  = False,
) -> str:
    fc_parts   = [f"[0:v]{vf_chain}[vout]"]
    af_orig    = _atempo_chain(speed_factor)
    fade_start = max(0.0, target_length - fade_out_sec)
    orig_vol   = "volume=0.3," if include_original else "volume=0,"
    fc_parts.append(
        f"[0:a]{af_orig},{orig_vol}"
        f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.1f}[aorig]"
    )
    fc_parts.append(
        f"[1:a]aloop=loop=-1:size=2e+09,"
        f"atrim=0:{target_length:.3f},"
        f"volume={music_volume:.3f},"
        f"afade=t=in:st=0:d=0.5,"
        f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.1f}[amusic]"
    )
    fc_parts.append(
        "[aorig][amusic]amix=inputs=2:duration=first:dropout_transition=2:weights='1 1'[aout]"
    )
    fc = ";".join(fc_parts)
    logger.debug(f"[FILTER] filter_complex built")
    return fc


def recommended_blur_frames(speed_factor: float) -> int:
    if speed_factor < 2.0: return 0
    if speed_factor < 3.0: return 3
    if speed_factor < 5.0: return 4
    if speed_factor < 8.0: return 6
    return 8


def _builtin_grade(preset: str) -> Optional[str]:
    presets = {
        "GoldenHour":   "eq=contrast=1.05:saturation=1.15:brightness=0.04,hue=s=1.1:h=5",
        "Misty":        "eq=contrast=0.92:saturation=0.85:brightness=0.06,hue=s=0.9",
        "Dramatic":     "eq=contrast=1.20:saturation=1.10:brightness=-0.04,hue=s=1.05",
        "Pastel":       "eq=contrast=0.90:saturation=0.75:brightness=0.08",
        "NightGlow":    "eq=contrast=1.10:saturation=0.70:brightness=-0.06,hue=s=0.6:h=200",
        "NaturalVivid": "eq=contrast=1.08:saturation=1.20:brightness=0.01",
    }
    return presets.get(preset)


def _gaussian_weights(n: int) -> str:
    if n <= 1:
        return "1"
    sigma   = (n - 1) / 4.0
    weights = [math.exp(-0.5 * ((i - (n - 1) / 2) / sigma) ** 2) for i in range(n)]
    total   = sum(weights)
    norm    = [w / total * n for w in weights]
    return " ".join(f"{w:.4f}" for w in norm)


def _atempo_chain(speed: float) -> str:
    stages: list = []
    rem = float(speed)
    while rem > 2.0 + 1e-6:
        stages.append("atempo=2.000000"); rem /= 2.0
    while rem < 0.5 - 1e-6:
        stages.append("atempo=0.500000"); rem /= 0.5
    stages.append(f"atempo={rem:.6f}")
    return ",".join(stages)
