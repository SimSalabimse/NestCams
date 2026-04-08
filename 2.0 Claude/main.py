#!/usr/bin/env python3
"""
Bird Motion Video Processor v2.0
OpenCL-accelerated motion detection + hardware video encoding.
"""

import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QCheckBox, QTextEdit, QTabWidget,
    QMessageBox, QLineEdit, QSlider, QTableWidget, QTableWidgetItem,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

from motion_detector import MotionDetector
from video_processor import VideoProcessor
from youtube_uploader import YouTubeUploader
from update_checker import UpdateChecker
from config_manager import ConfigManager
from analytics_dashboard import AnalyticsDashboard
from real_time_monitor import RealTimeMonitor
from bird_detector import get_bird_detector
from dark_mode import apply_dark_mode

# Fix Unicode logging on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bird_processor.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

import cv2
OPENCL_AVAILABLE = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()


class ProcessingThread(QThread):
    progress_update = pyqtSignal(int, str)
    time_estimate = pyqtSignal(float)
    processing_complete = pyqtSignal(bool, str, dict)

    def __init__(self, config, video_path, output_path, batch_mode=False):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.output_path = output_path
        self.batch_mode = batch_mode
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            t0 = time.time()
            self.progress_update.emit(5, "Analysing video for motion…")
            detector = MotionDetector(self.config)
            motion_segments, stats = detector.detect_motion(
                self.video_path,
                progress_callback=lambda p: self._emit(5, 45, p, "Detecting motion"),
            )

            if self.cancelled:
                self.processing_complete.emit(False, "Cancelled", {})
                return
            if not motion_segments:
                self.processing_complete.emit(False, "No motion detected in video", {})
                return

            self.progress_update.emit(50, "Processing video segments…")
            processor = VideoProcessor(self.config)

            if self.batch_mode:
                results = processor.create_timelapse_batch(
                    self.video_path, motion_segments, self.output_path,
                    progress_callback=lambda p: self._emit(50, 100, p, "Creating time-lapses"),
                    time_estimate_callback=self.time_estimate.emit,
                )
                if results:
                    msg = f"Created {len(results)} time-lapses:\n" + "\n".join(
                        f"  {k}s → {v}" for k, v in results.items()
                    )
                    stats["output_files"] = results
                    stats["processing_time"] = time.time() - t0
                    self.processing_complete.emit(True, msg, stats)
                else:
                    self.processing_complete.emit(False, "Failed to create time-lapses", {})
            else:
                ok = processor.create_timelapse(
                    self.video_path, motion_segments, self.output_path,
                    progress_callback=lambda p: self._emit(50, 100, p, "Creating time-lapse"),
                    time_estimate_callback=self.time_estimate.emit,
                )
                if ok:
                    stats["output_file"] = self.output_path
                    stats["processing_time"] = time.time() - t0
                    self.processing_complete.emit(True, f"Saved: {self.output_path}", stats)
                else:
                    self.processing_complete.emit(False, "Processing failed — check log", {})

        except Exception as e:
            logger.exception("Processing error")
            self.processing_complete.emit(False, f"Error: {e}", {})

    def _emit(self, lo, hi, pct, msg):
        if self.cancelled:
            return
        val = int(lo + (hi - lo) * (pct / 100))
        self.progress_update.emit(val, f"{msg}: {pct:.1f}%")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.analytics = AnalyticsDashboard()
        self.processing_thread = None
        self.real_time_monitor = None
        self.music_paths = {60: None, 600: None, 3600: None}
        self._init_ui()

    def _init_ui(self):
        accel = "OpenCL GPU" if OPENCL_AVAILABLE else "CPU"
        self.setWindowTitle(f"Bird Motion Video Processor v2.0 — {accel}")
        self.setGeometry(100, 100, 1000, 850)

        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        title = QLabel("Bird Box Motion Processor v2.0")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        if OPENCL_AVAILABLE:
            banner = QLabel("✅ OpenCL GPU acceleration enabled — fast motion detection active")
            banner.setStyleSheet("background:#16a34a;color:white;padding:8px;border-radius:4px;font-weight:bold;")
        else:
            banner = QLabel("ℹ️ CPU mode — update GPU drivers for OpenCL acceleration")
            banner.setStyleSheet("background:#d97706;color:white;padding:8px;border-radius:4px;")
        banner.setAlignment(Qt.AlignCenter)
        banner.setWordWrap(True)
        layout.addWidget(banner)

        tabs = QTabWidget()
        tabs.addTab(self._make_main_tab(), "Process Video")
        tabs.addTab(self._make_settings_tab(), "Settings")
        tabs.addTab(self._make_upload_tab(), "YouTube Upload")
        tabs.addTab(self._make_analytics_tab(), "Analytics")
        tabs.addTab(self._make_monitor_tab(), "Real-Time Monitor")
        tabs.addTab(self._make_updates_tab(), "Updates")
        layout.addWidget(tabs)

        self.statusBar().showMessage(f"Ready — {accel}")

    def _make_main_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Input
        grp = QGroupBox("Input Video")
        row = QHBoxLayout()
        self.input_label = QLabel("No file selected")
        row.addWidget(self.input_label)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_input)
        row.addWidget(btn)
        grp.setLayout(row)
        layout.addWidget(grp)

        # Output settings
        out_grp = QGroupBox("Output Settings")
        out_layout = QVBoxLayout()
        self.batch_check = QCheckBox("Batch mode — create 60s, 10min and 1hr videos at once")
        self.batch_check.stateChanged.connect(lambda s: self.len_widget.setEnabled(not s))
        out_layout.addWidget(self.batch_check)

        self.len_widget = QWidget()
        len_row = QHBoxLayout(self.len_widget)
        len_row.addWidget(QLabel("Target length:"))
        self.length_combo = QComboBox()
        self.length_combo.addItems(["60 seconds", "10 minutes", "1 hour", "Custom"])
        self.length_combo.currentIndexChanged.connect(
            lambda i: self.custom_length.setEnabled(i == 3)
        )
        len_row.addWidget(self.length_combo)
        self.custom_length = QSpinBox()
        self.custom_length.setRange(10, 7200)
        self.custom_length.setValue(300)
        self.custom_length.setSuffix(" s")
        self.custom_length.setEnabled(False)
        len_row.addWidget(self.custom_length)
        len_row.addStretch()
        out_layout.addWidget(self.len_widget)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Save to:"))
        self.output_label = QLabel("Default: same folder as input")
        path_row.addWidget(self.output_label)
        out_btn = QPushButton("Choose…")
        out_btn.clicked.connect(self._browse_output)
        path_row.addWidget(out_btn)
        out_layout.addLayout(path_row)
        out_grp.setLayout(out_layout)
        layout.addWidget(out_grp)

        # Music
        music_grp = QGroupBox("Background Music")
        music_layout = QVBoxLayout()
        self.add_music_check = QCheckBox("Add background music")
        music_layout.addWidget(self.add_music_check)
        for length, label in [(60, "60s"), (600, "10min"), (3600, "1hr")]:
            row2 = QHBoxLayout()
            row2.addWidget(QLabel(f"{label} video:"))
            lbl = QLabel("No music selected")
            setattr(self, f"music_lbl_{length}", lbl)
            row2.addWidget(lbl)
            btn2 = QPushButton("Browse…")
            btn2.clicked.connect(lambda _, l=length: self._browse_music(l))
            row2.addWidget(btn2)
            music_layout.addLayout(row2)
        music_grp.setLayout(music_layout)
        layout.addWidget(music_grp)

        # Quick Settings
        qs_grp = QGroupBox("Quick Settings")
        qs_layout = QVBoxLayout()

        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Motion sensitivity:"))
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(1, 10)
        self.sens_slider.setValue(5)
        self.sens_slider.setTickPosition(QSlider.TicksBelow)
        self.sens_slider.setTickInterval(1)
        sens_row.addWidget(self.sens_slider)
        self.sens_label = QLabel("Medium (5)")
        self.sens_slider.valueChanged.connect(
            lambda v: self.sens_label.setText(f"{'Low' if v <= 3 else 'High' if v >= 8 else 'Medium'} ({v})")
        )
        sens_row.addWidget(self.sens_label)
        qs_layout.addLayout(sens_row)

        self.motion_blur_check = QCheckBox("Motion blur (smoother fast-forward)")
        self.motion_blur_check.setChecked(True)
        qs_layout.addWidget(self.motion_blur_check)

        self.strong_blur_check = QCheckBox("Strong motion blur (slower but smoother)")
        self.strong_blur_check.setChecked(False)
        qs_layout.addWidget(self.strong_blur_check)

        self.color_correction_check = QCheckBox("Color correction")
        self.color_correction_check.setChecked(True)
        qs_layout.addWidget(self.color_correction_check)

        qs_grp.setLayout(qs_layout)
        layout.addWidget(qs_grp)

        # Buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Processing")
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("✕  Cancel")
        self.cancel_btn.setFont(QFont("Segoe UI", 12))
        self.cancel_btn.setMinimumHeight(50)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # Progress
        prog_grp = QGroupBox("Progress")
        prog_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        prog_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        prog_layout.addWidget(self.progress_label)
        self.eta_label = QLabel("")
        prog_layout.addWidget(self.eta_label)
        prog_grp.setLayout(prog_layout)
        layout.addWidget(prog_grp)

        # Log
        log_grp = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(130)
        log_layout.addWidget(self.log_box)
        log_grp.setLayout(log_layout)
        layout.addWidget(log_grp)

        return tab

    def _make_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        motion_grp = QGroupBox("Motion Detection")
        motion_layout = QVBoxLayout()

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Min motion duration:"))
        self.min_dur = QDoubleSpinBox()
        self.min_dur.setRange(0.1, 10.0)
        self.min_dur.setValue(0.5)
        self.min_dur.setSuffix(" s")
        self.min_dur.setSingleStep(0.1)
        r1.addWidget(self.min_dur)
        r1.addStretch()
        motion_layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Segment padding:"))
        self.padding = QDoubleSpinBox()
        self.padding.setRange(0.0, 10.0)
        self.padding.setValue(1.0)
        self.padding.setSuffix(" s")
        self.padding.setSingleStep(0.5)
        r2.addWidget(self.padding)
        r2.addStretch()
        motion_layout.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Frame skip (1=every frame):"))
        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 10)
        self.frame_skip.setValue(2)
        r3.addWidget(self.frame_skip)
        r3.addStretch()
        motion_layout.addLayout(r3)

        motion_grp.setLayout(motion_layout)
        layout.addWidget(motion_grp)

        proc_grp = QGroupBox("Encoding")
        proc_layout = QVBoxLayout()

        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("CPU threads:"))
        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, os.cpu_count() or 8)
        self.cpu_threads.setValue(max(1, (os.cpu_count() or 4) - 1))
        t_row.addWidget(self.cpu_threads)
        t_row.addWidget(QLabel(f"(max {os.cpu_count()})"))
        t_row.addStretch()
        proc_layout.addLayout(t_row)

        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Output quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low (faster)", "Medium", "High", "Maximum (slower)"])
        self.quality_combo.setCurrentIndex(2)
        q_row.addWidget(self.quality_combo)
        q_row.addStretch()
        proc_layout.addLayout(q_row)

        proc_grp.setLayout(proc_layout)
        layout.addWidget(proc_grp)

        layout.addStretch()
        return tab

    def _make_upload_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("YouTube Upload (coming soon)"))
        return tab

    def _make_analytics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Analytics Dashboard (coming soon)"))
        return tab

    def _make_monitor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Real-Time Monitor (coming soon)"))
        return tab

    def _make_updates_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Updates (coming soon)"))
        return tab

    # Event handlers
    def _browse_input(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if p:
            self.input_label.setText(p)
            self._log(f"Input: {p}")

    def _browse_output(self):
        p, _ = QFileDialog.getSaveFileName(self, "Output Location", "", "MP4 (*.mp4)")
        if p:
            if not p.lower().endswith(".mp4"):
                p += ".mp4"
            self.output_label.setText(p)

    def _browse_music(self, length: int):
        p, _ = QFileDialog.getOpenFileName(
            self, f"Music for {length}s video", "",
            "Audio (*.mp3 *.wav *.aac *.m4a);;All Files (*)"
        )
        if p:
            self.music_paths[length] = p
            self.add_music_check.setChecked(True)
            getattr(self, f"music_lbl_{length}").setText(Path(p).name)

    def _start(self):
        in_path = self.input_label.text()
        if in_path == "No file selected" or not os.path.exists(in_path):
            QMessageBox.warning(self, "Error", "Please select a valid input video.")
            return

        out_path = self.output_label.text()
        if out_path == "Default: same folder as input":
            out_path = str(Path(in_path).parent / (Path(in_path).stem + "_timelapse"))

        config = self._build_config()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self._log("Starting…")

        self.processing_thread = ProcessingThread(
            config, in_path, out_path, self.batch_check.isChecked()
        )
        self.processing_thread.progress_update.connect(self._on_progress)
        self.processing_thread.time_estimate.connect(self._on_eta)
        self.processing_thread.processing_complete.connect(self._on_done)
        self.processing_thread.start()

    def _cancel(self):
        if self.processing_thread and self.processing_thread.isRunning():
            if QMessageBox.question(self, "Cancel", "Cancel processing?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self.processing_thread.cancel()
                self._log("Cancelling…")

    def _on_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_label.setText(msg)
        self._log(msg)

    def _on_eta(self, secs):
        if secs > 0:
            eta = datetime.now() + timedelta(seconds=secs)
            self.eta_label.setText(f"ETA: {int(secs//60)}m {int(secs%60)}s ({eta.strftime('%H:%M:%S')})")
        else:
            self.eta_label.setText("")

    def _on_done(self, ok, msg, stats):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100 if ok else 0)
        self.eta_label.setText("")
        self._log(msg)
        if ok:
            QMessageBox.information(self, "Done", msg)
        else:
            QMessageBox.warning(self, "Failed", msg)

    def _build_config(self) -> dict:
        if self.batch_check.isChecked():
            lengths = [60, 600, 3600]
        else:
            idx = self.length_combo.currentIndex()
            lengths = [[60], [600], [3600], [self.custom_length.value()]][idx]

        return {
            "sensitivity": self.sens_slider.value(),
            "min_motion_duration": self.min_dur.value(),
            "segment_padding": self.padding.value(),
            "frame_skip": self.frame_skip.value(),
            "use_gpu": True,
            "cpu_threads": self.cpu_threads.value(),
            "quality": self.quality_combo.currentIndex(),
            "add_music": self.add_music_check.isChecked(),
            "music_paths": self.music_paths,
            "target_lengths": lengths,
            "motion_blur": self.motion_blur_check.isChecked(),
            "blur_strength": "strong" if self.strong_blur_check.isChecked() else "light",
            "color_correction": self.color_correction_check.isChecked(),
        }

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")
        logger.info(msg)


def main():
    app = QApplication(sys.argv)
    apply_dark_mode(app)
    window = MainWindow()
    window.show()

    logger.info("=" * 60)
    if OPENCL_AVAILABLE:
        logger.info("🚀 OpenCL GPU acceleration active")
    else:
        logger.info("ℹ️ CPU mode — OpenCL not available")
    logger.info("=" * 60)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()