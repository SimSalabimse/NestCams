"""
batch_processor.py — Process multiple 24-hour videos in one batch with resume support.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from motion_detector  import MotionDetector
from video_processor  import VideoProcessor
from metadata_handler import MetadataHandler
from utils            import log_session

logger = logging.getLogger(__name__)

_LOCK_EXT  = ".processing.lock"
_DONE_EXT  = ".done.json"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


class BatchProcessor:
    """
    Processes all video files in a folder with:
      • Crash-safe lock files
      • Per-file .done.json sidecars (skip finished files on resume)
      • cancel_flag support
      • Optional progress_callback(current_file_index, total_files, pct_this_file)
    """

    def __init__(self, config: dict):
        self.config       = config
        self._cancel_flag = threading.Event()
        self._meta        = MetadataHandler()

    def set_cancel_flag(self, flag: threading.Event) -> None:
        self._cancel_flag = flag

    def process_folder(
        self,
        input_folder:  str,
        output_folder: str,
        durations:     List[int]          = None,
        resume_from:   Optional[str]      = None,
        progress_cb:   Optional[Callable] = None,
        status_cb:     Optional[Callable] = None,
    ) -> dict:
        if durations is None:
            durations = [59, 720, 3600]

        os.makedirs(output_folder, exist_ok=True)
        files = self._collect_files(input_folder)

        if resume_from:
            files = self._filter_unfinished(files)
            logger.info(f"[BATCH] Resume mode: {len(files)} remaining")

        total   = len(files)
        results = {"total": total, "success": 0, "failed": 0, "skipped": 0, "files": []}
        log_session(f"Batch start: {total} files → {output_folder}")

        for idx, fpath in enumerate(files):
            if self._cancel_flag.is_set():
                logger.info("[BATCH] Cancelled")
                break

            done = fpath + _DONE_EXT
            lock = fpath + _LOCK_EXT

            if os.path.exists(done):
                results["skipped"] += 1
                continue

            try: Path(lock).write_text("processing")
            except OSError: pass

            if status_cb:
                status_cb(f"Processing {Path(fpath).name} ({idx+1}/{total})")

            def _pcb(pct, _i=idx, _t=total):
                if progress_cb: progress_cb(_i, _t, pct)

            file_result = self._process_one(fpath, output_folder, durations, _pcb)
            results["files"].append(file_result)

            if file_result["success"]:
                results["success"] += 1
                with open(done, "w") as fh:
                    json.dump(file_result, fh, indent=2, default=str)
            else:
                results["failed"] += 1

            try: os.remove(lock)
            except OSError: pass

        log_session(
            f"Batch done: success={results['success']} "
            f"failed={results['failed']} skipped={results['skipped']}"
        )
        return results

    def resume_unfinished(self, folder: str) -> List[str]:
        return [str(p).replace(_LOCK_EXT, "") for p in Path(folder).glob(f"*{_LOCK_EXT}")]

    def _process_one(self, fpath, output_folder, durations, progress_cb) -> dict:
        t0     = time.time()
        result = {"file": fpath, "success": False, "outputs": [], "error": None, "elapsed": 0}
        try:
            detector = MotionDetector(self.config)
            detector.set_cancel_flag(self._cancel_flag)
            segments, stats = detector.detect_motion(fpath)

            if not segments:
                result["error"] = "No motion detected"
                return result

            base = Path(fpath).stem
            for dur in durations:
                if self._cancel_flag.is_set():
                    break
                label    = self._dur_label(dur)
                out_path = os.path.join(output_folder, f"{base}_{label}.mp4")
                proc     = VideoProcessor(dict(self.config))
                proc.set_cancel_flag(self._cancel_flag)
                ok = proc.create_timelapse(fpath, segments, out_path,
                                           target_length=dur, progress_callback=progress_cb)
                if ok:
                    result["outputs"].append(out_path)
                    self._meta.generate_best_thumbnail(out_path)
                    self._meta.embed_metadata(out_path, {**stats, "target_duration": dur})

            result["success"] = bool(result["outputs"])
        except Exception as exc:
            logger.exception(f"[BATCH] Error on {fpath}: {exc}")
            result["error"] = str(exc)
        finally:
            result["elapsed"] = round(time.time() - t0, 1)
        return result

    @staticmethod
    def _collect_files(folder: str) -> List[str]:
        return sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")
        )

    @staticmethod
    def _filter_unfinished(files: List[str]) -> List[str]:
        return [f for f in files if not os.path.exists(f + _DONE_EXT)]

    @staticmethod
    def _dur_label(sec: int) -> str:
        if sec < 120:  return f"{sec}s"
        if sec < 3600: return f"{sec//60}min"
        return f"{sec//3600}h"
