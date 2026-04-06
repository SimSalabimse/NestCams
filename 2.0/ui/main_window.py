import sys
import os
import time
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QProgressBar,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
    QTextEdit,
    QLineEdit,
    QSizePolicy,
)
from typing import Tuple
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont
import cv2
from core.video_processor import VideoProcessor
from core.motion_detector import MotionDetector
from core.hardware_manager import HardwareManager
from utils.config import Config
from utils.logger import logger
from utils.github_updater import GitHubUpdater
from utils.overall_eta import OverallProcessETA


class ProcessingThread(QThread):
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    eta_update = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        processor,
        input_path,
        output_path,
        duration,
        add_music,
        music_path,
        mode,
        quality,
        min_segment_duration,
        output_format,
        frame_subsample=5,
        motion_buffer=2.0,
    ):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_path = output_path
        self.duration = duration
        self.add_music = add_music
        self.music_path = music_path
        self.mode = mode  # "fast" or "quality"
        self.quality = quality  # "high", "medium", "low"
        self.min_segment_duration = min_segment_duration  # seconds
        self.output_format = output_format  # "mp4", "webm", "avi"
        self.frame_subsample = frame_subsample  # analyze every Nth frame
        self.motion_buffer = motion_buffer  # seconds before/after motion
        self.cancel_flag = False
        self.eta_tracker = OverallProcessETA(
            console=False, update_callback=self._emit_eta
        )

    def _emit_eta(self, eta_text: str):
        self.eta_update.emit(eta_text)

    def cancel(self):
        """Request cancellation of processing"""
        self.cancel_flag = True

    def on_motion_detection_progress(self, current_frame, total_frames):
        """Callback for motion detection progress (5% to 35% of total)"""
        if total_frames > 0:
            frame_progress = (current_frame / total_frames) * 100
            # Scale frame progress from 5% to 35% range
            overall_progress = 5 + (frame_progress * 0.3)
            self.progress.emit(int(overall_progress))

    def on_timelapse_progress(self, current_value, max_value):
        """Callback for time-lapse progress (35% to 70% of total)"""
        # For timelapse, current_value and max_value are typically segment counts
        # Scale from 35% to 70% range
        if max_value > 0:
            segment_progress = (current_value / max_value) * 100
            overall_progress = 35 + (segment_progress * 0.35)
        else:
            overall_progress = 35
        self.progress.emit(int(overall_progress))

    def run(self):
        try:
            self.status_update.emit("Step 1/4: Detecting motion in video...")
            self.progress.emit(5)
            segments = self.processor.extract_motion_segments(
                self.input_path,
                "",
                progress_callback=self.on_motion_detection_progress,
                cancel_flag=self,
                min_segment_duration=self.min_segment_duration,
                frame_subsample=self.frame_subsample,
                motion_buffer=self.motion_buffer,
                eta_tracker=self.eta_tracker,
            )
            total_motion_time = sum(end - start for start, end in segments)
            self.status_update.emit(
                f"Motion detection complete. Found {len(segments)} motion segments ({total_motion_time:.1f}s total)."
            )
            self.progress.emit(35)

            if self.cancel_flag:
                return

            self.status_update.emit("Step 2/4: Creating time-lapse...")
            temp_output = self.output_path.replace(".mp4", "_temp.mp4")
            self.processor.create_timelapse(
                self.input_path,
                segments,
                self.duration,
                temp_output,
                mode=self.mode,
                quality=self.quality,
                progress_callback=self.on_timelapse_progress,
                cancel_flag=self,
                output_format=self.output_format,
                eta_tracker=self.eta_tracker,
            )
            self.status_update.emit("Time-lapse creation complete.")
            self.progress.emit(70)

            if self.cancel_flag:
                return

            if self.add_music and self.music_path:
                self.status_update.emit("Step 3/4: Adding music to video...")
                self.processor.add_music(temp_output, self.music_path, self.output_path)
                os.remove(temp_output)
                self.status_update.emit("Music added successfully.")
            else:
                self.status_update.emit("Step 3/4: Finalizing video...")
                os.rename(temp_output, self.output_path)
            self.progress.emit(90)

            self.status_update.emit("Step 4/4: Processing complete!")
            self.progress.emit(100)
            if hasattr(self, "eta_tracker"):
                self.eta_tracker.start_next_phase("Final cleanup")
                self.eta_tracker.finish()
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.hardware = HardwareManager()
        self.motion_detector = MotionDetector(
            sensitivity=self.config.get("motion_sensitivity"),
            min_area=self.config.get("min_area"),
            algorithm=self.config.get("motion_algorithm"),
            use_gpu=self.config.get("use_gpu", False),
        )
        self.processor = VideoProcessor(self.motion_detector)
        self.updater = GitHubUpdater(self.config.get("github_repo"))
        self.thread = None

        self.init_ui()
        self.check_for_updates()

    def init_ui(self):
        self.setWindowTitle("NestCams 2.0 - Video Processor")
        self.resize(980, 700)
        self.setMinimumSize(900, 640)

        self.setStyleSheet(
            "QWidget { background: #f5f7fa; color: #1f2937; font-family: 'Segoe UI', Arial, sans-serif; }"
            "QGroupBox { border: 1px solid #d1d5db; border-radius: 14px; margin-top: 18px; background: #ffffff; }"
            "QGroupBox:title { subcontrol-origin: margin; left: 14px; padding: 0 8px 0 8px; color: #111827; font-weight: 700; }"
            "QLabel { font-size: 12px; }"
            "QPushButton { min-height: 38px; font-weight: 600; border-radius: 10px; background: #2563eb; color: #ffffff; padding: 0 16px; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:disabled { background: #93c5fd; color: #f8fafc; }"
            "QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { min-height: 36px; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0 10px; background: #ffffff; }"
            "QProgressBar { border: 1px solid #cbd5e1; border-radius: 16px; background: #e2e8f0; color: #111827; min-height: 34px; }"
            "QProgressBar::chunk { border-radius: 16px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #0ea5e9); }"
            "QTextEdit { border: 1px solid #cbd5e1; border-radius: 10px; background: #ffffff; }"
            "QScrollBar:vertical { width: 10px; background: #f1f5f9; border-radius: 5px; }"
            "QScrollBar::handle:vertical { background: #cbd5e1; border-radius: 5px; }"
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # File selection
        file_group = QGroupBox("Input/Output")
        file_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.input_label = QLabel("No input file selected")
        input_btn = QPushButton("Select Video")
        input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(input_btn)
        file_layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output will be saved to: output/")
        self.output_label.setWordWrap(True)
        output_btn = QPushButton("Select Output Location")
        output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(output_btn)
        file_layout.addLayout(output_layout)
        file_group.setLayout(file_layout)
        self.output_dir = self.config.get("output_dir")
        self.output_label.setText(f"Output will be saved to: {self.output_dir}/")
        layout.addWidget(file_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        # Processing mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Processing Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Fast (Recommended)", "Quality (Compatible)"])
        self.mode_combo.setToolTip(
            "Fast: Uses FFmpeg filters for speed\nQuality: Extracts segments individually for maximum compatibility"
        )
        mode_layout.addWidget(self.mode_combo)
        settings_layout.addLayout(mode_layout)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Time-lapse Duration:"))
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(
            [
                "30 seconds",
                "60 seconds",
                "5 minutes",
                "10 minutes",
                "30 minutes",
                "1 hour",
            ]
        )
        # Set default to 60 seconds
        self.duration_combo.setCurrentText("60 seconds")
        duration_layout.addWidget(self.duration_combo)
        settings_layout.addLayout(duration_layout)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Output Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["High", "Medium", "Low"])
        self.quality_combo.setCurrentText("High")
        quality_layout.addWidget(self.quality_combo)
        settings_layout.addLayout(quality_layout)

        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Motion Sensitivity:"))
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.1, 1.0)
        self.sensitivity_spin.setValue(self.config.get("motion_sensitivity"))
        self.sensitivity_spin.setSingleStep(0.1)
        sensitivity_layout.addWidget(self.sensitivity_spin)
        settings_layout.addLayout(sensitivity_layout)

        self.smoothing_check = QCheckBox("Enable smoothing")
        self.smoothing_check.setChecked(self.config.get("smoothing"))
        settings_layout.addWidget(self.smoothing_check)

        self.gpu_check = QCheckBox("Use GPU acceleration (if available)")
        self.gpu_check.setChecked(self.config.get("use_gpu", False))
        self.gpu_check.setToolTip(
            "Enable GPU-assisted preprocessing and motion detection when supported by OpenCV."
        )
        settings_layout.addWidget(self.gpu_check)

        music_layout = QHBoxLayout()
        self.music_check = QCheckBox("Add music")
        self.music_check.stateChanged.connect(self.toggle_music_selection)
        music_layout.addWidget(self.music_check)
        self.music_btn = QPushButton("Select Music File")
        self.music_btn.clicked.connect(self.select_music)
        self.music_btn.setEnabled(False)
        music_layout.addWidget(self.music_btn)
        settings_layout.addLayout(music_layout)

        update_layout = QHBoxLayout()
        self.update_check = QCheckBox("Enable update checking")
        self.update_check.setChecked(self.config.get("update_check", False))
        self.update_check.stateChanged.connect(self.toggle_update_check)
        update_layout.addWidget(self.update_check)

        self.repo_input = QLineEdit(self.config.get("github_repo", ""))
        self.repo_input.setPlaceholderText("owner/repo")
        self.repo_input.editingFinished.connect(self.save_settings)
        update_layout.addWidget(self.repo_input)
        settings_layout.addLayout(update_layout)

        self.update_button = QPushButton("Check for Updates Now")
        self.update_button.clicked.connect(self.check_for_updates)
        settings_layout.addWidget(self.update_button)

        # Advanced settings
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()

        # Motion detection algorithm
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Motion Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(
            ["MOG2 (Recommended)", "CNT (Fastest)", "KNN", "Simple Difference"]
        )
        self.algo_combo.setToolTip(
            "MOG2: Best for complex scenes\nCNT: Fastest for stable backgrounds\nKNN: Good for stable backgrounds\nSimple: Fast but less accurate"
        )
        algo_layout.addWidget(self.algo_combo)
        advanced_layout.addLayout(algo_layout)

        # Minimum segment duration
        segment_layout = QHBoxLayout()
        segment_layout.addWidget(QLabel("Min Segment Duration:"))
        self.segment_spin = QDoubleSpinBox()
        self.segment_spin.setRange(0.1, 5.0)
        self.segment_spin.setValue(0.5)
        self.segment_spin.setSingleStep(0.1)
        self.segment_spin.setSuffix(" seconds")
        segment_layout.addWidget(self.segment_spin)
        advanced_layout.addLayout(segment_layout)

        # Frame subsample rate
        subsample_layout = QHBoxLayout()
        subsample_layout.addWidget(QLabel("Frame Subsample:"))
        self.subsample_spin = QSpinBox()
        self.subsample_spin.setRange(1, 30)
        self.subsample_spin.setValue(5)
        self.subsample_spin.setSingleStep(1)
        self.subsample_spin.setSuffix(" (analyze every Nth frame)")
        self.subsample_spin.setToolTip(
            "Higher values = faster processing but may miss motion"
        )
        subsample_layout.addWidget(self.subsample_spin)
        advanced_layout.addLayout(subsample_layout)

        # Motion buffer
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Motion Buffer:"))
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0.0, 10.0)
        self.buffer_spin.setValue(2.0)
        self.buffer_spin.setSingleStep(0.5)
        self.buffer_spin.setSuffix(" seconds")
        self.buffer_spin.setToolTip("Time before/after motion to include in segments")
        buffer_layout.addWidget(self.buffer_spin)
        advanced_layout.addLayout(buffer_layout)

        # Output format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP4 (H.264)", "WebM (VP9)", "AVI (XVID)"])
        format_layout.addWidget(self.format_combo)
        advanced_layout.addLayout(format_layout)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Progress and status display
        self.status_label = QLabel("Ready to process video")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #666; font-size: 12px; padding: 6px; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;"
        )
        layout.addWidget(self.status_label)

        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(12)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimumWidth(320)
        self.progress_bar.setMaximumWidth(840)
        self.progress_bar.setMinimumHeight(36)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar, 2)

        self.progress_label = QLabel("0%")
        self.progress_label.setMinimumWidth(70)
        self.progress_label.setMinimumHeight(36)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet(
            "font-weight: bold; color: #1d4ed8; font-size: 14px; background-color: #eef2ff; padding: 8px 10px; border-radius: 8px;"
        )
        progress_layout.addWidget(self.progress_label)

        self.eta_label = QLabel("ETA: --:--")
        self.eta_label.setMinimumWidth(180)
        self.eta_label.setMinimumHeight(36)
        self.eta_label.setAlignment(Qt.AlignCenter)
        self.eta_label.setStyleSheet(
            "font-size: 13px; color: #111827; background-color: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0;"
        )
        progress_layout.addWidget(self.eta_label)

        layout.addLayout(progress_layout)

        control_layout = QHBoxLayout()
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_btn)

        self.youtube_btn = QPushButton("Upload to YouTube")
        self.youtube_btn.clicked.connect(self.upload_to_youtube)
        self.youtube_btn.setEnabled(False)
        control_layout.addWidget(self.youtube_btn)

        layout.addLayout(control_layout)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        self.show()

        # Load saved settings
        self.load_settings()

    def select_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.input_path = file_path
            self.input_label.setText(os.path.basename(file_path))
            self.process_btn.setEnabled(True)
            logger.info(f"Selected input file: {file_path}")

    def select_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f"Output will be saved to: {dir_path}/")
            self.config.set("output_dir", dir_path)

    def select_music(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Music File", "", "Audio Files (*.mp3 *.wav *.m4a)"
        )
        if file_path:
            self.music_path = file_path
            logger.info(f"Selected music file: {file_path}")

    def toggle_music_selection(self, state):
        self.music_btn.setEnabled(state == 2)  # Qt.CheckState.Checked

    def toggle_update_check(self, state):
        self.config.set("update_check", state == 2)

    def save_settings(self):
        mode_map = {"Fast (Recommended)": "fast", "Quality (Compatible)": "quality"}
        quality_map = {"High": "high", "Medium": "medium", "Low": "low"}
        algo_map = {
            "MOG2 (Recommended)": "mog2",
            "CNT (Fastest)": "cnt",
            "KNN": "knn",
            "Simple Difference": "simple",
        }
        format_map = {"MP4 (H.264)": "mp4", "WebM (VP9)": "webm", "AVI (XVID)": "avi"}

        self.config.set("motion_sensitivity", self.sensitivity_spin.value())
        self.config.set("smoothing", self.smoothing_check.isChecked())
        self.config.set("use_gpu", self.gpu_check.isChecked())
        self.config.set("github_repo", self.repo_input.text().strip())
        self.config.set(
            "processing_mode", mode_map.get(self.mode_combo.currentText(), "fast")
        )
        self.config.set(
            "output_quality", quality_map.get(self.quality_combo.currentText(), "high")
        )
        self.config.set(
            "motion_algorithm", algo_map.get(self.algo_combo.currentText(), "mog2")
        )
        self.config.set("min_segment_duration", self.segment_spin.value())
        self.config.set("frame_subsample", self.subsample_spin.value())
        self.config.set("motion_buffer", self.buffer_spin.value())
        self.config.set(
            "output_format", format_map.get(self.format_combo.currentText(), "mp4")
        )
        self.updater = GitHubUpdater(self.repo_input.text().strip())

    def update_motion_detector(self, algorithm: str):
        """Recreate the motion detector with the selected algorithm and sensitivity."""
        self.motion_detector = MotionDetector(
            sensitivity=self.sensitivity_spin.value(),
            min_area=self.config.get("min_area"),
            algorithm=algorithm,
            use_gpu=self.gpu_check.isChecked(),
        )
        self.processor.motion_detector = self.motion_detector

    def load_settings(self):
        """Load saved settings into UI controls"""
        # Processing mode
        mode_reverse_map = {
            "fast": "Fast (Recommended)",
            "quality": "Quality (Compatible)",
        }
        saved_mode = self.config.get("processing_mode", "fast")
        self.mode_combo.setCurrentText(
            mode_reverse_map.get(saved_mode, "Fast (Recommended)")
        )

        # Output quality
        quality_reverse_map = {"high": "High", "medium": "Medium", "low": "Low"}
        saved_quality = self.config.get("output_quality", "high")
        self.quality_combo.setCurrentText(
            quality_reverse_map.get(saved_quality, "High")
        )

        # Motion algorithm
        algo_reverse_map = {
            "mog2": "MOG2 (Recommended)",
            "cnt": "CNT (Fastest)",
            "knn": "KNN",
            "simple": "Simple Difference",
        }
        saved_algo = self.config.get("motion_algorithm", "mog2")
        self.algo_combo.setCurrentText(
            algo_reverse_map.get(saved_algo, "MOG2 (Recommended)")
        )

        # Output format
        format_reverse_map = {
            "mp4": "MP4 (H.264)",
            "webm": "WebM (VP9)",
            "avi": "AVI (XVID)",
        }
        saved_format = self.config.get("output_format", "mp4")
        self.format_combo.setCurrentText(
            format_reverse_map.get(saved_format, "MP4 (H.264)")
        )

        # Other settings
        self.segment_spin.setValue(self.config.get("min_segment_duration", 0.5))
        self.subsample_spin.setValue(self.config.get("frame_subsample", 5))
        self.buffer_spin.setValue(self.config.get("motion_buffer", 2.0))
        self.gpu_check.setChecked(self.config.get("use_gpu", False))

    def check_motion_time(
        self, input_path: str, target_duration: int
    ) -> Tuple[bool, float]:
        """Check if video has sufficient motion time. Returns (has_sufficient_motion, total_motion_time)"""
        try:
            segments = self.processor.extract_motion_segments(
                input_path,
                "",
                min_segment_duration=self.segment_spin.value(),
                frame_subsample=self.subsample_spin.value(),
                motion_buffer=self.buffer_spin.value(),
            )
            total_motion_time = sum(end - start for start, end in segments)
            return total_motion_time >= target_duration, total_motion_time
        except Exception as e:
            logger.error(f"Failed to check motion time: {e}")
            return True, 0  # Assume sufficient to allow processing to continue

    def start_processing(self):
        if not hasattr(self, "input_path"):
            QMessageBox.warning(self, "Error", "Please select an input video file.")
            return

        duration_map = {
            "30 seconds": 30,
            "60 seconds": 60,
            "5 minutes": 300,
            "10 minutes": 600,
            "30 minutes": 1800,
            "1 hour": 3600,
        }
        duration = duration_map[self.duration_combo.currentText()]
        duration_label = self.duration_combo.currentText()

        mode_map = {"Fast (Recommended)": "fast", "Quality (Compatible)": "quality"}
        mode = mode_map[self.mode_combo.currentText()]

        quality_map = {"High": "high", "Medium": "medium", "Low": "low"}
        quality = quality_map[self.quality_combo.currentText()]

        algo_map = {
            "MOG2 (Recommended)": "mog2",
            "CNT (Fastest)": "cnt",
            "KNN": "knn",
            "Simple Difference": "simple",
        }
        algorithm = algo_map[self.algo_combo.currentText()]

        min_segment_duration = self.segment_spin.value()
        frame_subsample = self.subsample_spin.value()
        motion_buffer = self.buffer_spin.value()

        self.update_motion_detector(algorithm)

        format_map = {"MP4 (H.264)": "mp4", "WebM (VP9)": "webm", "AVI (XVID)": "avi"}
        output_format = format_map[self.format_combo.currentText()]

        format_extensions = {"mp4": ".mp4", "webm": ".webm", "avi": ".avi"}
        extension = format_extensions.get(output_format, ".mp4")
        output_name = f"timelapse_{duration}s{extension}"
        output_path = os.path.join(self.config.get("output_dir"), output_name)
        self.save_settings()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting processing...")
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.youtube_btn.setEnabled(False)

        self.processing_start_time = time.time()
        self.eta_label.setText("ETA: calculating...")
        self.thread = ProcessingThread(
            self.processor,
            self.input_path,
            output_path,
            duration,
            self.music_check.isChecked(),
            getattr(self, "music_path", None),
            mode,
            quality,
            min_segment_duration,
            output_format,
            frame_subsample=frame_subsample,
            motion_buffer=motion_buffer,
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.status_update.connect(self.update_status)
        self.thread.eta_update.connect(self.update_eta)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def cancel_processing(self):
        """Cancel the current video processing"""
        if self.thread and self.thread.isRunning():
            logger.info("Processing cancelled by user")
            self.thread.cancel()
            self.status_label.setText("✗ Processing cancelled.")
            self.eta_label.setText("ETA: --:--")
            self.status_label.setStyleSheet("color: #ffc107; font-size: 11px;")
            self.cancel_btn.setEnabled(False)
            self.process_btn.setEnabled(True)
            self.youtube_btn.setEnabled(False)

    def update_progress(self, value: int):
        """Update progress bar and percentage label."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"{value}%")

    def update_eta(self, eta_text: str):
        """Update the ETA label with the latest estimated time remaining."""
        self.eta_label.setText(eta_text)

    def _format_seconds(self, seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def update_status(self, status_text: str):
        """Update the status label with current processing step"""
        self.status_label.setText(status_text)
        logger.info(status_text)

    def processing_finished(self, output_path):
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.youtube_btn.setEnabled(True)
        self.output_path = output_path
        self.status_label.setText("✓ Processing complete. Ready for next video.")
        self.eta_label.setText("ETA: 00:00")
        self.status_label.setStyleSheet("color: #28a745; font-size: 11px;")
        QMessageBox.information(
            self, "Success", f"Video processed successfully!\nSaved to: {output_path}"
        )
        logger.info(f"Processing completed: {output_path}")

    def processing_error(self, error_msg):
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.youtube_btn.setEnabled(False)
        self.eta_label.setText("ETA: --:--")
        self.status_label.setText(f"✗ Error: {error_msg}")
        self.status_label.setStyleSheet("color: #dc3545; font-size: 11px;")
        QMessageBox.critical(self, "Error", f"Processing failed: {error_msg}")
        logger.error(f"Processing error: {error_msg}")

    def upload_to_youtube(self):
        # Placeholder for YouTube upload
        QMessageBox.information(
            self, "YouTube Upload", "YouTube upload functionality to be implemented."
        )
        logger.info("YouTube upload initiated")

    def check_for_updates(self):
        if not self.update_check.isChecked():
            logger.info("Update check disabled")
            return

        repo = self.repo_input.text().strip()
        if not repo:
            logger.info("No GitHub repo configured for updates")
            return

        self.updater = GitHubUpdater(repo)
        try:
            has_update, latest_version, url = self.updater.check_for_updates("2.0.0")
            if has_update:
                reply = QMessageBox.question(
                    self,
                    "Update Available",
                    f"A new version ({latest_version}) is available. Would you like to download it?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    import webbrowser

                    webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Update check failed: {e}")
            QMessageBox.information(
                self,
                "Update Check",
                "Unable to check for updates. Please verify your GitHub repository and internet connection.",
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
