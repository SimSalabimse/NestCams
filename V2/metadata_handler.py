"""
metadata_handler.py — Embeds stats metadata and generates thumbnails/reports.
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MetadataHandler:
    """
    Handles:
      • Embedding processing metadata into output .mp4 via FFmpeg -metadata
      • Writing a companion .json sidecar
      • Generating a 1280×720 thumbnail from the best frame
      • Creating an HTML activity report
    """

    def embed_metadata(self, video_path: str, stats: dict) -> bool:
        try:
            sidecar = Path(video_path).with_suffix(".json")
            with open(sidecar, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2, default=str)

            tmp = video_path + ".meta_tmp.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-c", "copy", "-map_metadata", "0",
                "-metadata", f"comment=BirdBoxProcessor|{json.dumps(stats, default=str)[:500]}",
                "-metadata", "encoded_by=BirdBoxVideoProcessor",
                "-metadata", f"date={datetime.now().strftime('%Y-%m-%d')}",
                tmp,
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode == 0:
                os.replace(tmp, video_path)
                logger.info(f"[META] Embedded: {video_path}")
                return True
            logger.warning(f"[META] Embed failed: {r.stderr.decode()[:300]}")
            try: os.remove(tmp)
            except OSError: pass
            return False
        except Exception as exc:
            logger.exception(f"[META] embed_metadata error: {exc}")
            return False

    def generate_thumbnail(
        self,
        video_path:    str,
        timestamp_sec: float = 3.0,
        output_path:   Optional[str] = None,
        size:          tuple = (1280, 720),
    ) -> Optional[str]:
        try:
            if output_path is None:
                output_path = str(Path(video_path).with_suffix(".thumb.jpg"))
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None
            frame_r = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(output_path, frame_r, [cv2.IMWRITE_JPEG_QUALITY, 92])
            logger.info(f"[META] Thumbnail: {output_path}")
            return output_path
        except Exception as exc:
            logger.exception(f"[META] generate_thumbnail error: {exc}")
            return None

    def generate_best_thumbnail(
        self, video_path: str, n_candidates: int = 8,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        from frame_analyzer import FrameAnalyzer
        fa    = FrameAnalyzer()
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return self.generate_thumbnail(video_path, 3.0, output_path)

        candidates = []
        for i in range(n_candidates):
            pos = int(total * (i + 0.5) / n_candidates)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok:
                candidates.append(frame)
        cap.release()

        best = fa.find_best_frame(candidates)
        if best is None:
            return self.generate_thumbnail(video_path, 3.0, output_path)
        if output_path is None:
            output_path = str(Path(video_path).with_suffix(".thumb.jpg"))
        resized = cv2.resize(best, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return output_path

    def create_activity_report(
        self,
        segments:     list,
        motion_stats: dict,
        output_path:  Optional[str] = None,
    ) -> str:
        total_dur  = motion_stats.get("duration", 0)
        motion_dur = motion_stats.get("motion_duration", 0)
        n_segs     = motion_stats.get("motion_segments", 0)
        pct        = motion_dur / max(total_dur, 1) * 100
        now        = datetime.now().strftime("%Y-%m-%d %H:%M")

        rows = ""
        for i, (s, e) in enumerate(segments[:200]):
            rows += (f"<tr><td>{i+1}</td><td>{self._fmt_time(s)}</td>"
                     f"<td>{self._fmt_time(e)}</td><td>{e-s:.2f}s</td></tr>")

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Bird Activity Report — {now}</title>
<style>
  body{{font-family:Segoe UI,sans-serif;background:#111;color:#eee;padding:20px}}
  h1{{color:#f59e0b}} table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #333;padding:6px 10px;text-align:left}}
  th{{background:#1f2937}} tr:nth-child(even){{background:#1a1a2e}}
  .stat{{display:inline-block;background:#1f2937;border-radius:8px;
          padding:12px 20px;margin:8px;min-width:160px}}
  .num{{font-size:2em;font-weight:bold;color:#f59e0b}}
</style></head><body>
<h1>🐦 Bird Activity Report</h1>
<p>Generated: {now}</p>
<div>
  <div class="stat"><div class="num">{n_segs}</div>Motion events</div>
  <div class="stat"><div class="num">{motion_dur:.1f}s</div>Total motion</div>
  <div class="stat"><div class="num">{pct:.1f}%</div>Activity %</div>
  <div class="stat"><div class="num">{total_dur/3600:.1f}h</div>Duration</div>
</div>
<h2>Event Timeline</h2>
<table><tr><th>#</th><th>Start</th><th>End</th><th>Duration</th></tr>
{rows}</table></body></html>"""

        if output_path:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html

    @staticmethod
    def _fmt_time(sec: float) -> str:
        h, r = divmod(int(sec), 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
