"""
social_media_exporter.py — Platform-specific export, caption, and hashtag generation.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PLATFORM_SPECS: Dict[str, dict] = {
    "YouTubeShorts":  {"width": 1080, "height": 1920, "max_sec": 60,  "fps": 30, "suffix": "_yt_short"},
    "TikTok":         {"width": 1080, "height": 1920, "max_sec": 60,  "fps": 30, "suffix": "_tiktok"},
    "InstagramReels": {"width": 1080, "height": 1920, "max_sec": 90,  "fps": 30, "suffix": "_ig_reel"},
    "InstagramSquare":{"width": 1080, "height": 1080, "max_sec": 60,  "fps": 30, "suffix": "_ig_sq"},
    "Facebook":       {"width": 1080, "height": 1350, "max_sec": 240, "fps": 30, "suffix": "_fb"},
    "X":              {"width": 1280, "height": 720,  "max_sec": 140, "fps": 30, "suffix": "_x"},
    "YouTube":        {"width": 1920, "height": 1080, "max_sec": 3600,"fps": 30, "suffix": "_yt"},
}

HASHTAG_SETS: Dict[str, List[str]] = {
    "BirdLovers":   ["#BirdLovers","#BirdWatching","#GardenBirds","#BackyardBirds","#NatureLovers","#BirdsOfInstagram","#FeatheredFriends"],
    "NatureASMR":   ["#NatureASMR","#ASMR","#NatureSounds","#RelaxingNature","#MindfulNature","#CalmingVideos","#NatureHeals"],
    "NestCamDaily": ["#NestCam","#BirdNest","#NestBox","#BabyBirds","#NestCamDaily","#WildlifeCamera","#NatureDiary"],
    "ViralWildlife":["#ViralWildlife","#NatureIsAmazing","#WildlifeLovers","#AnimalVideos","#NatureVideo","#WildlifePhotography","#CuteAnimals"],
    "BirdTok":      ["#BirdTok","#BirdTikTok","#FYP","#ForYou","#Wildlife","#NatureTok","#BirdVideo"],
}

CAPTIONS: Dict[str, List[str]] = {
    "Minimal":   ["Nature at work 🪺","Life in the nest box 🐦","A day in the life 🌿","Watch closely… 👀"],
    "FunFacts":  [
        "Did you know? Most songbirds incubate eggs for 12–14 days 🥚",
        "Fun fact: Baby birds can gain up to 10× their hatch weight in a week! 🐣",
        "Nature fact: Both parents feed nestlings up to 400 times a day! 🐦",
    ],
    "StoryMode": [
        "Day {day} in the nest box — things are getting exciting! 🪺 Drop a ❤️ if you're following along!",
        "The family is growing! Watch this incredible moment from our garden nest box 🐦🌿",
        "You won't believe what happened in our nest box today! 🐣👀 Watch till the end!",
    ],
}


class SocialMediaExporter:
    def __init__(self, ffmpeg: str = "ffmpeg"):
        self._ff = ffmpeg

    def export_for_platform(
        self, video_path: str, platform: str,
        config: dict, output_dir: Optional[str] = None,
    ) -> Optional[str]:
        spec = PLATFORM_SPECS.get(platform)
        if not spec:
            logger.error(f"[EXPORT] Unknown platform: {platform}")
            return None
        if output_dir is None:
            output_dir = str(Path(video_path).parent)
        stem = Path(video_path).stem
        out  = os.path.join(output_dir, stem + spec["suffix"] + ".mp4")
        w, h = spec["width"], spec["height"]
        vf   = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,fps={spec['fps']}")
        cmd  = [
            self._ff, "-y", "-i", video_path,
            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-t", str(spec["max_sec"]), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            if r.returncode == 0:
                logger.info(f"[EXPORT] {platform} → {out}")
                return out
            logger.error(f"[EXPORT] FFmpeg error: {r.stderr.decode(errors='replace')[:400]}")
            return None
        except Exception as exc:
            logger.exception(f"[EXPORT] {platform}: {exc}")
            return None

    def export_social_pack(
        self, video_path: str, platforms: List[str],
        config: dict, output_dir: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        return {p: self.export_for_platform(video_path, p, config, output_dir) for p in platforms}

    def generate_caption(self, stats: dict, style: str = "FunFacts", day: int = 1) -> str:
        import random
        templates  = CAPTIONS.get(style, CAPTIONS["Minimal"])
        base       = random.choice(templates).format(day=day)
        motion_pct = stats.get("motion_duration", 0) / max(stats.get("duration", 1), 1) * 100
        n_segs     = stats.get("motion_segments", 0)
        date       = stats.get("date", "today")
        return base + f"\n\n📊 {n_segs} activity events · {motion_pct:.0f}% motion · {date}"

    def generate_hashtags(
        self, set_name: str = "BirdLovers",
        detected_species: List[str] = None,
        custom_tags: List[str] = None,
        max_tags: int = 20,
    ) -> List[str]:
        tags = list(HASHTAG_SETS.get(set_name, HASHTAG_SETS["BirdLovers"]))
        if detected_species:
            tags += [f"#{sp.replace(' ','')}" for sp in detected_species]
        if custom_tags:
            tags += custom_tags
        seen: set = set()
        unique: List[str] = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower()); unique.append(t)
        return unique[:max_tags]

    def get_platform_specs(self) -> Dict[str, dict]:
        return dict(PLATFORM_SPECS)
