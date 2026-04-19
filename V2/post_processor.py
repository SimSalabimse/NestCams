"""
post_processor.py — FFmpeg filter-chain builder for the post-concat stage.

Correct VF order (must never change):
    setpts-reset → deflicker → eq → hqdn3d → tmix(blur) →
    setpts(speedup) → film_grain → vignette → lut3d/grade →
    transpose(Shorts) → drawtext(watermark)
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class PostProcessor:
    """Builds -vf and -af strings for the single FFmpeg post-processing pass."""

    def build_video_filter(
        self,
        speed_factor:       float = 1.0,
        is_short:           bool  = False,
        deflicker_strength: int   = 5,
        motion_blur_frames: int   = 0,
        contrast:           float = 1.0,
        brightness:         float = 0.0,
        saturation:         float = 1.0,
        denoise:            bool  = True,
        watermark:          Optional[str] = None,
        film_grain:         float = 0.0,
        vignette_strength:  float = 0.0,
        lut_path:           Optional[str] = None,
        color_grade:        Optional[str] = None,
    ) -> str:
        parts: list = []

        # 1. Reset PTS drift
        parts.append("setpts=PTS-STARTPTS")

        # 2. Deflicker
        if deflicker_strength > 0:
            parts.append(f"deflicker=size={deflicker_strength}:mode=am")

        # 3. eq colour grade
        if contrast != 1.0 or saturation != 1.0 or brightness != 0.0:
            parts.append(
                f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}:brightness={brightness:.3f}"
            )

        # 4. Denoise
        if denoise:
            parts.append("hqdn3d=luma_spatial=1.5:luma_tmp=2.5:chroma_spatial=1:chroma_tmp=2")

        # 5. Motion blur (tmix) — BEFORE speedup
        if motion_blur_frames >= 2:
            weights = self._gaussian_weights(motion_blur_frames)
            parts.append(f"tmix=frames={motion_blur_frames}:weights='{weights}'")

        # 6. Speedup — AFTER blur
        if speed_factor > 1.001:
            parts.append(f"setpts=PTS/{speed_factor:.6f}")

        # 7. Film grain
        if film_grain > 0.001:
            parts.append(f"noise=alls={int(film_grain * 100)}:allf=t+u")

        # 8. Vignette
        if vignette_strength > 0.01:
            angle = math.pi / (2.0 + (0.6 - min(0.6, vignette_strength)) * 3)
            parts.append(f"vignette=angle={angle:.4f}:mode=forward")

        # 9. LUT / grade
        if lut_path:
            safe = lut_path.replace("\\", "/").replace("'", r"\'")
            parts.append(f"lut3d=file='{safe}'")
        elif color_grade:
            grade_filter = self._builtin_grade(color_grade)
            if grade_filter:
                parts.append(grade_filter)

        # 10. Rotate for Short
        if is_short:
            parts.append("transpose=1")

        # 11. Watermark
        if watermark:
            safe = watermark.replace("'", r"\'").replace(":", r"\:")
            parts.append(
                f"drawtext=text='{safe}':fontcolor=white@0.8:fontsize=22"
                ":x=10:y=10:shadowcolor=black@0.6:shadowx=1:shadowy=1"
            )

        vf = ",".join(parts)
        logger.debug(f"[PP] vf chain ({len(parts)} filters)")
        return vf

    def build_audio_filter(
        self,
        speed_factor:     float = 1.0,
        target_length:    float = 60.0,
        include_original: bool  = True,
        fade_out_sec:     float = 2.0,
    ) -> str:
        parts = [self._atempo_chain(speed_factor)]
        if include_original:
            parts.append("volume=0.3")
        fade_start = max(0.0, target_length - fade_out_sec)
        parts.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.1f}")
        return ",".join(parts)

    @staticmethod
    def _atempo_chain(speed: float) -> str:
        stages: list = []
        rem = float(speed)
        while rem > 2.0 + 1e-6:
            stages.append("atempo=2.000000"); rem /= 2.0
        while rem < 0.5 - 1e-6:
            stages.append("atempo=0.500000"); rem /= 0.5
        stages.append(f"atempo={rem:.6f}")
        return ",".join(stages)

    @staticmethod
    def _gaussian_weights(n: int) -> str:
        if n <= 1:
            return "1"
        sigma   = (n - 1) / 4.0
        weights = [math.exp(-0.5 * ((i - (n - 1) / 2) / sigma) ** 2) for i in range(n)]
        total   = sum(weights)
        norm    = [w / total * n for w in weights]
        return " ".join(f"{w:.4f}" for w in norm)

    @staticmethod
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
