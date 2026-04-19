"""
template_manager.py — Load, save, and manage JSON look templates.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BUILTIN_TEMPLATES: Dict[str, dict] = {
    "Baby Birds First Flight": {
        "description": "Warm, soft look for fledgling moments",
        "color_grade_preset": "GoldenHour", "film_grain": 0.08, "vignette_strength": 0.25,
        "contrast": 1.05, "saturation": 1.12, "brightness": 0.03,
        "deflicker_size": 5, "motion_blur_frames": -1, "pacing_mode": "Cinematic",
        "hashtag_set": "BirdLovers", "caption_style": "StoryMode",
    },
    "Rainy Day Drama": {
        "description": "Moody high-contrast look for wet weather",
        "color_grade_preset": "Dramatic", "film_grain": 0.15, "vignette_strength": 0.40,
        "contrast": 1.18, "saturation": 0.90, "brightness": -0.04,
        "deflicker_size": 5, "motion_blur_frames": 4, "pacing_mode": "Cinematic",
        "hashtag_set": "NatureASMR", "caption_style": "FunFacts",
    },
    "Golden Hour Magic": {
        "description": "Warm sunset tones for evening activity",
        "color_grade_preset": "GoldenHour", "film_grain": 0.05, "vignette_strength": 0.20,
        "contrast": 1.08, "saturation": 1.20, "brightness": 0.05,
        "deflicker_size": 4, "motion_blur_frames": 3, "pacing_mode": "Cinematic",
        "hashtag_set": "BirdTok", "caption_style": "Minimal",
    },
    "Funny Moments": {
        "description": "Fast-cut high energy for comedic moments",
        "color_grade_preset": "NaturalVivid", "film_grain": 0.0, "vignette_strength": 0.0,
        "contrast": 1.10, "saturation": 1.15, "brightness": 0.02,
        "deflicker_size": 3, "motion_blur_frames": 0, "pacing_mode": "FastCut",
        "hashtag_set": "ViralWildlife", "caption_style": "FunFacts",
    },
    "Misty Morning": {
        "description": "Soft desaturated look for dawn footage",
        "color_grade_preset": "Misty", "film_grain": 0.10, "vignette_strength": 0.30,
        "contrast": 0.95, "saturation": 0.85, "brightness": 0.06,
        "deflicker_size": 6, "motion_blur_frames": 3, "pacing_mode": "Cinematic",
        "hashtag_set": "NatureASMR", "caption_style": "Minimal",
    },
    "Night Glow": {
        "description": "Dark IR-camera look with cool tones",
        "color_grade_preset": "NightGlow", "film_grain": 0.20, "vignette_strength": 0.50,
        "contrast": 1.12, "saturation": 0.65, "brightness": -0.05,
        "deflicker_size": 7, "motion_blur_frames": 4, "pacing_mode": "Cinematic",
        "hashtag_set": "NestCamDaily", "caption_style": "StoryMode",
    },
    "Nature ASMR": {
        "description": "Slow meditative pacing with natural audio",
        "color_grade_preset": "NaturalVivid", "film_grain": 0.0, "vignette_strength": 0.10,
        "contrast": 1.02, "saturation": 1.08, "brightness": 0.01,
        "deflicker_size": 5, "motion_blur_frames": -1, "pacing_mode": "StoryDriven",
        "hashtag_set": "NatureASMR", "caption_style": "FunFacts",
    },
    "Viral Short": {
        "description": "Optimised for TikTok/Shorts max engagement",
        "color_grade_preset": "Dramatic", "film_grain": 0.05, "vignette_strength": 0.15,
        "contrast": 1.15, "saturation": 1.20, "brightness": 0.02,
        "deflicker_size": 4, "motion_blur_frames": -1, "pacing_mode": "FastCut",
        "hashtag_set": "BirdTok", "caption_style": "StoryMode",
    },
}


class TemplateManager:
    def __init__(self, templates_path: str = "templates.json"):
        self._path    = templates_path
        self._builtin = dict(BUILTIN_TEMPLATES)
        self._user:   Dict[str, dict] = {}
        self._load_user_templates()

    def list_templates(self) -> List[str]:
        return list(self._builtin.keys()) + [k for k in self._user if k not in self._builtin]

    def get(self, name: str) -> Optional[dict]:
        return self._user.get(name) or self._builtin.get(name)

    def save_user_template(self, name: str, settings: dict) -> None:
        self._user[name] = dict(settings)
        self._write_user_templates()
        logger.info(f"[TMPL] Saved: {name}")

    def delete_user_template(self, name: str) -> bool:
        if name in self._builtin:
            return False
        if name in self._user:
            del self._user[name]
            self._write_user_templates()
            return True
        return False

    def apply_to_config(self, name: str, config: dict) -> dict:
        tmpl = self.get(name)
        if tmpl is None:
            return config
        return {**config, **tmpl}

    def export_template(self, name: str, output_path: str) -> bool:
        tmpl = self.get(name)
        if tmpl is None:
            return False
        try:
            with open(output_path, "w") as fh:
                json.dump({"name": name, "template": tmpl}, fh, indent=2)
            return True
        except Exception as exc:
            logger.error(f"[TMPL] Export failed: {exc}")
            return False

    def import_template(self, json_path: str) -> Optional[str]:
        try:
            with open(json_path) as fh:
                data = json.load(fh)
            name = data.get("name", Path(json_path).stem)
            tmpl = data.get("template", data)
            self.save_user_template(name, tmpl)
            return name
        except Exception as exc:
            logger.error(f"[TMPL] Import failed: {exc}")
            return None

    def _load_user_templates(self) -> None:
        try:
            if os.path.exists(self._path):
                with open(self._path) as fh:
                    self._user = json.load(fh)
        except Exception as exc:
            logger.warning(f"[TMPL] Load failed: {exc}")
            self._user = {}

    def _write_user_templates(self) -> None:
        try:
            with open(self._path, "w") as fh:
                json.dump(self._user, fh, indent=2)
        except Exception as exc:
            logger.error(f"[TMPL] Write failed: {exc}")
