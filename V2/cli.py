#!/usr/bin/env python3
"""
cli.py — Headless command-line interface for Bird Box Video Processor.

Usage examples:
  python cli.py --input nest.mp4 --duration 60
  python cli.py --input nest.mp4 --duration 60 720 3600 --music track.mp3
  python cli.py --input nest.mp4 --duration 60 --roi 100,100,400,300
  python cli.py --batch /recordings --output /outputs --duration 60 720
  python cli.py --input nest.mp4 --autotune --duration 60
"""

import argparse
import json
import logging
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_detector  import MotionDetector
from video_processor  import VideoProcessor
from batch_processor  import BatchProcessor
from auto_tuner       import AutoTuner
from metadata_handler import MetadataHandler
from utils            import log_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="birdbox-cli",
        description="Bird Box Video Processor — headless mode",
    )
    g_io = p.add_argument_group("Input / Output")
    g_io.add_argument("--input",    help="Path to a single input video")
    g_io.add_argument("--batch",    help="Path to folder for batch processing")
    g_io.add_argument("--output",   help="Output folder (default: same as input)")
    g_io.add_argument("--duration", nargs="+", type=int, default=[59], metavar="SEC")
    g_io.add_argument("--music",    help="Path to music file")

    g_m = p.add_argument_group("Motion Detection")
    g_m.add_argument("--sensitivity", type=int, default=5, choices=range(1, 11), metavar="1-10")
    g_m.add_argument("--roi",         help="ROI as x,y,w,h pixels e.g. 100,100,640,480")
    g_m.add_argument("--frame-skip",  type=int, default=2, dest="frame_skip")
    g_m.add_argument("--mog2",        action="store_true")
    g_m.add_argument("--autotune",    action="store_true")

    g_q = p.add_argument_group("Video Quality")
    g_q.add_argument("--quality",    choices=["Low","Medium","High","Maximum"], default="High")
    g_q.add_argument("--no-gpu",     action="store_true", dest="no_gpu")
    g_q.add_argument("--deflicker",  type=int, default=5, dest="deflicker_size")
    g_q.add_argument("--no-denoise", action="store_true", dest="no_denoise")
    g_q.add_argument("--contrast",   type=float, default=1.0)
    g_q.add_argument("--saturation", type=float, default=1.0)
    g_q.add_argument("--brightness", type=float, default=0.0)
    g_q.add_argument("--watermark",  help="Watermark text")
    g_q.add_argument("--grade",      help="Colour grade: GoldenHour/Misty/Dramatic/Pastel/NightGlow")

    g_b = p.add_argument_group("Batch")
    g_b.add_argument("--resume",   action="store_true")
    g_b.add_argument("--settings", help="Path to settings.json")

    g_s = p.add_argument_group("Social Export")
    g_s.add_argument("--platforms", nargs="*",
                     choices=["YouTubeShorts","TikTok","InstagramReels","Facebook","X"])
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if not args.input and not args.batch:
        parser.print_help()
        return 1

    config: dict = {}
    if args.settings and os.path.exists(args.settings):
        with open(args.settings) as fh:
            config = json.load(fh)

    config.update({
        "sensitivity":      args.sensitivity,
        "frame_skip":       args.frame_skip,
        "use_mog2":         args.mog2,
        "quality":          args.quality,
        "use_gpu":          not args.no_gpu,
        "deflicker_size":   args.deflicker_size,
        "denoise":          not args.no_denoise,
        "contrast":         args.contrast,
        "saturation":       args.saturation,
        "brightness":       args.brightness,
        "watermark_text":   args.watermark,
        "color_grade_preset": args.grade or "None",
    })

    if args.roi:
        parts = [int(v.strip()) for v in args.roi.split(",")]
        if len(parts) == 4:
            config["_pixel_roi"] = parts

    cancel_flag = threading.Event()

    if args.autotune and args.input:
        logger.info("Running AutoTuner…")
        tuner     = AutoTuner(config)
        suggested = tuner.suggest_optimal_settings(
            args.input,
            progress_callback=lambda p: print(f"\r  AutoTune {p:.0f}%", end="", flush=True),
        )
        print()
        config.update(suggested)
        logger.info(f"AutoTune result: {suggested}")

    if args.batch:
        out_dir = args.output or args.batch
        proc    = BatchProcessor(config)
        proc.set_cancel_flag(cancel_flag)

        def _pcb(idx, total, pct):
            print(f"\r  [{idx+1}/{total}] {pct:.0f}%", end="", flush=True)

        results = proc.process_folder(
            args.batch, out_dir,
            durations   = args.duration,
            resume_from = args.resume or None,
            progress_cb = _pcb,
            status_cb   = lambda m: logger.info(m),
        )
        print()
        logger.info(f"Batch done: {results['success']} ok / {results['failed']} failed / {results['skipped']} skipped")
        return 0

    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"Input not found: {args.input}")
            return 2

        out_dir = args.output or os.path.dirname(os.path.abspath(args.input))

        if "_pixel_roi" in config:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(args.input)
            w   = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            px, py, pw, ph = config.pop("_pixel_roi")
            config["roi"] = {"x": px/w, "y": py/h, "w": pw/w, "h": ph/h}

        logger.info(f"Detecting motion: {args.input}")
        detector = MotionDetector(config)
        detector.set_cancel_flag(cancel_flag)
        segments, stats = detector.detect_motion(
            args.input,
            progress_callback=lambda p: print(f"\r  Detecting {p:.0f}%", end="", flush=True),
        )
        print()

        if not segments:
            logger.error("No motion detected.")
            return 3

        logger.info(f"Motion: {len(segments)} segs / {stats.get('motion_duration',0):.1f}s")

        meta   = MetadataHandler()
        all_ok = True

        for dur in args.duration:
            label    = _dur_label(dur)
            base     = os.path.splitext(os.path.basename(args.input))[0]
            out_path = os.path.join(out_dir, f"{base}_{label}.mp4")
            os.makedirs(out_dir, exist_ok=True)

            logger.info(f"Generating {label} → {out_path}")
            proc = VideoProcessor(config)
            proc.set_cancel_flag(cancel_flag)
            ok   = proc.create_timelapse(
                args.input, segments, out_path,
                target_length=dur, music_path=args.music,
                progress_callback=lambda p: print(f"\r  Encoding {p:.0f}%", end="", flush=True),
            )
            print()

            if ok:
                meta.embed_metadata(out_path, {**stats, "target_duration": dur})
                meta.generate_best_thumbnail(out_path)
                logger.info(f"✅  {out_path}")
            else:
                logger.error(f"❌  Failed: {out_path}")
                all_ok = False

        if args.platforms and all_ok:
            from social_media_exporter import SocialMediaExporter
            exporter = SocialMediaExporter()
            for dur in args.duration:
                label    = _dur_label(dur)
                base     = os.path.splitext(os.path.basename(args.input))[0]
                src_path = os.path.join(out_dir, f"{base}_{label}.mp4")
                if os.path.exists(src_path):
                    for platform in args.platforms:
                        result = exporter.export_for_platform(src_path, platform, config, out_dir)
                        if result:
                            logger.info(f"✅  {platform}: {result}")

        return 0 if all_ok else 4

    return 0


def _dur_label(sec: int) -> str:
    if sec < 120:  return f"{sec}s"
    if sec < 3600: return f"{sec//60}min"
    return f"{sec//3600}h"


if __name__ == "__main__":
    sys.exit(main())
