import time
import sys
from typing import Callable, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class OverallProcessETA:
    def __init__(
        self,
        total_phases: int = 4,
        detection_weight: float = 0.85,
        console: bool = True,
        update_callback: Optional[Callable[[str], None]] = None,
    ):
        self.start_time = time.time()
        self.phases: List[Tuple[str, float]] = []
        self.current_phase = 0
        self.detection_weight = detection_weight
        self.segments_total = 0
        self.segments_done = 0
        self.last_print = 0.0
        self.console = console
        self.update_callback = update_callback
        self.detection_pbar = None
        self.detection_total = 0
        self.detection_current = 0

    def add_phase(self, name: str, weight: Optional[float] = None):
        if weight is None:
            weight = (1.0 - self.detection_weight) / max(1, len(self.phases) + 1)
        self.phases.append((name, weight))
        self.current_phase = min(self.current_phase, len(self.phases) - 1)
        self._notify_status()

    def start_detection(self, total_effective_frames: int):
        self.add_phase("Motion detection", self.detection_weight)
        self.detection_total = total_effective_frames
        self.detection_current = 0
        if self.console and tqdm is not None:
            self.detection_pbar = tqdm(
                total=total_effective_frames,
                desc="Detection",
                unit="frames",
                ncols=90,
                smoothing=0.1,
                file=sys.stdout,
            )
        self._notify_status()

    def update_detection(self, steps: int = 1):
        if self.detection_pbar is not None:
            self.detection_pbar.update(steps)
        self.detection_current = min(
            self.detection_total, self.detection_current + steps
        )
        self._notify_status()

    def finish_detection(self, num_segments_found: int):
        if self.detection_pbar is not None:
            self.detection_pbar.close()
            self.detection_pbar = None
        self.segments_total = num_segments_found
        self.current_phase = min(self.current_phase + 1, len(self.phases))
        self._notify_status(f"Found {num_segments_found} motion segments")

    def start_next_phase(self, name: str, weight: Optional[float] = None):
        self.add_phase(name, weight)
        self.current_phase = len(self.phases) - 1
        self._notify_status()

    def finish_segment(self):
        self.segments_done += 1
        self._notify_status()

    def finish(self):
        elapsed = time.time() - self.start_time
        self._notify_status("Completed")
        if self.console:
            print(f"\n✅ Process completed in {elapsed/60:.1f} minutes")

    def _compute_status(self) -> Tuple[float, str, float]:
        elapsed = time.time() - self.start_time
        progress = 0.0
        for i, (name, weight) in enumerate(self.phases):
            if i < self.current_phase:
                progress += weight
            elif i == self.current_phase:
                if name == "Motion detection":
                    if self.detection_pbar is not None:
                        progress += weight * (
                            self.detection_pbar.n / max(1, self.detection_pbar.total)
                        )
                    elif self.detection_total > 0:
                        progress += weight * (
                            self.detection_current / self.detection_total
                        )
                    else:
                        progress += 0.0
                elif "segment" in name.lower() and self.segments_total > 0:
                    progress += weight * (self.segments_done / self.segments_total)
                else:
                    progress += 0.0
        if progress > 0.0:
            total_est = elapsed / progress
            remaining = max(0.0, total_est - elapsed)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            remaining = 0.0
            eta_str = "calculating..."
        return progress, eta_str, elapsed

    def _get_current_phase_name(self) -> str:
        if self.current_phase < len(self.phases):
            return self.phases[self.current_phase][0]
        return "Finishing"

    def _notify_status(self, extra_msg: str = ""):
        if time.time() - self.last_print < 0.3:
            return
        self.last_print = time.time()
        progress, eta_str, elapsed = self._compute_status()
        phase_name = self._get_current_phase_name()
        bar = f"[{int(progress * 100):3d}%] {phase_name:22} | ETA: {eta_str} | Elapsed: {elapsed/60:.1f} min"
        if extra_msg:
            bar += f" | {extra_msg}"
        if self.segments_total > 0:
            bar += f" | Segments: {self.segments_done}/{self.segments_total}"
        if self.console:
            print(f"\r{bar}", end="", flush=True)
        if self.update_callback:
            self.update_callback(bar)
