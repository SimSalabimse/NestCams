"""
caption_generator.py — Rule-based caption generator with optional LLM stub.
Supports English, German, and Spanish.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "BirdLovers": [
            "Watch this tiny bird build a masterpiece nest 🪺 #BirdTok",
            "Nature is incredible — look what happened in our nest box today! 🐦",
            "This little one has been busy all day 🌿 #NestCam",
            "You won't believe what's happening in our garden! 🥚 Watch till the end!",
            "Drop a ❤️ if this made your day! 🐣 #BirdLovers",
        ],
        "NatureASMR": [
            "Sit back and relax 🌿 Pure nature sounds from our nest box 🎶",
            "No music, no voiceover — just the sounds of nature 🌱 #NatureASMR",
            "Close your eyes and listen 🍃 #NatureASMR",
        ],
        "NestCamDaily": [
            "Day {day} update from the nest box! 🪺 Things are getting interesting…",
            "Incredible footage from our garden nest cam today 🐦🌿",
            "Can you believe what these birds got up to today? 👀 #NestCamDaily",
        ],
        "ViralWildlife": [
            "Wildlife in my garden is WILD 😂🐦 Watch this!",
            "I set up a nest cam and THIS happened 🤯 #ViralWildlife",
            "Nature never fails to surprise me 🌿💚 Share if you love birds!",
        ],
    },
    "de": {
        "BirdLovers": [
            "Schau dir an, was heute in unserer Nistkiste passiert ist! 🐦🪺",
            "Die Natur ist unglaublich — einfach zuschauen und staunen! 🌿",
            "Gib ein ❤️, wenn dich das glücklich macht! 🐣 #Vogelliebhaber",
        ],
    },
    "es": {
        "BirdLovers": [
            "¡Mira lo que pasó en nuestra caja nido hoy! 🐦🪺",
            "¡La naturaleza es increíble — no te lo puedes perder! 🌿",
            "¡Dale ❤️ si esto te alegró el día! 🐣 #AmorPorLasBirds",
        ],
    },
}

BIRD_FACTS = [
    "Most songbirds incubate eggs for 12–14 days 🥚",
    "Baby birds can gain up to 10× their hatch weight in the first week 🐣",
    "Both parents take turns feeding nestlings up to 400 times a day! 🐦",
    "A clutch of blue tit eggs can weigh more than the mother 💪",
    "Blackbirds can sing over 100 distinct melodies 🎶",
    "Some birds remove faecal sacs from the nest to keep it clean 🧹",
    "Nest boxes attract birds that might otherwise struggle to find cavities 🏡",
    "Fledglings leave the nest before they can fully fly 🌿",
]


class CaptionGenerator:
    def __init__(self, language: str = "en", llm_enabled: bool = False):
        self.language    = language
        self.llm_enabled = llm_enabled
        self._llm_client = None

    def generate(
        self,
        stats:         dict,
        hashtag_set:   str           = "BirdLovers",
        caption_style: str           = "BirdLovers",
        day:           int           = 1,
        hashtags:      List[str]     = None,
        platform:      str           = "YouTube",
        extra_context: Optional[str] = None,
    ) -> str:
        if self.llm_enabled and self._llm_client:
            return self._llm_caption(stats, extra_context, platform)
        return self._rule_based(stats, hashtag_set, caption_style, day, hashtags)

    def random_bird_fact(self) -> str:
        return random.choice(BIRD_FACTS)

    def _rule_based(self, stats, hashtag_set, style, day, hashtags) -> str:
        lang_pack = TEMPLATES.get(self.language, TEMPLATES["en"])
        templates = lang_pack.get(style, lang_pack.get("BirdLovers", []))
        if not templates:
            templates = TEMPLATES["en"]["BirdLovers"]

        base       = random.choice(templates).format(day=day)
        motion_pct = (stats.get("motion_duration", 0) / max(stats.get("duration", 1), 1) * 100)
        n_segs     = stats.get("motion_segments", 0)
        date       = stats.get("date", datetime.now().strftime("%Y-%m-%d"))
        stats_line = f"\n\n📊 {n_segs} activity events | {motion_pct:.0f}% in motion | {date}"

        fact_line = ""
        if random.random() < 0.5:
            fact_line = f"\n\n💡 Did you know? {self.random_bird_fact()}"

        tag_str = ""
        if hashtags:
            tag_str = "\n\n" + " ".join(hashtags[:20])

        return base + stats_line + fact_line + tag_str

    def _llm_caption(self, stats, extra_context, platform) -> str:
        """Stub for LLM-based caption generation. Wire up Anthropic/OpenAI API here."""
        logger.info("[CAPTION] LLM stub called — falling back to rule-based")
        return self._rule_based(stats, "BirdLovers", "BirdLovers", 1, None)

    def set_llm_client(self, client) -> None:
        self._llm_client = client
        self.llm_enabled = True
