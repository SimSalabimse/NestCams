#!/usr/bin/env python3
"""
Bird Motion Video Processor
A comprehensive video processing tool for extracting motion from long recordings
and creating time-lapse videos with adjustable length.
"""

import sys
import os
import json
import logging
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QProgressBar,
                             QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QCheckBox, QTextEdit, QTabWidget,
                             QMessageBox, QLineEdit, QSlider)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon

from motion_detector import MotionDetector
from video_processor import VideoProcessor
from youtube_uploader import YouTubeUploader
from update_checker import UpdateChecker
from config_manager import ConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bird_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Background thread for video processing to keep UI responsive"""
    progress_update = pyqtSignal(int, str)
    processing_complete = pyqtSignal(bool, str)
    
    def __init__(self, config, video_path, output_path):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.output_path = output_path
        
    def run(self):
        try:
            # Step 1: Detect motion
            self.progress_update.emit(10, "Analyzing video for motion...")
            detector = MotionDetector(self.config)
            motion_segments = detector.detect_motion(
                self.video_path,
                progress_callback=lambda p: self.progress_update.emit(10 + int(p * 0.4), f"Detecting motion: {p:.1f}%")
            )
            
            if not motion_segments:
                self.processing_complete.emit(False, "No motion detected in video")
                return
            
            # Step 2: Process video
            self.progress_update.emit(50, "Processing video segments...")
            processor = VideoProcessor(self.config)
            success = processor.create_timelapse(
                self.video_path,
                motion_segments,
                self.output_path,
                progress_callback=lambda p: self.progress_update.emit(50 + int(p * 0.5), f"Creating time-lapse: {p:.1f}%")
            )
            
            if success:
                self.processing_complete.emit(True, f"Video saved to: {self.output_path}")
            else:
                self.processing_complete.emit(False, "Failed to process video")
                
        except Exception as e:
            logger.exception("Error during processing")
            self.processing_complete.emit(False, f"Error: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.processing_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Bird Motion Video Processor v1.0")
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("Bird Box Motion Processor")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_main_tab(), "Process Video")
        tabs.addTab(self.create_settings_tab(), "Settings")
        tabs.addTab(self.create_upload_tab(), "YouTube Upload")
        tabs.addTab(self.create_updates_tab(), "Updates")
        layout.addWidget(tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_main_tab(self):
        """Main processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input file selection
        input_group = QGroupBox("Input Video")
        input_layout = QHBoxLayout()
        self.input_path_label = QLabel("No file selected")
        input_layout.addWidget(self.input_path_label)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(browse_btn)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        # Target length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Target Length:"))
        self.target_length = QComboBox()
        self.target_length.addItems(["60 seconds", "10 minutes", "1 hour", "Custom"])
        self.target_length.currentIndexChanged.connect(self.on_target_length_changed)
        length_layout.addWidget(self.target_length)
        
        self.custom_length = QSpinBox()
        self.custom_length.setRange(10, 7200)
        self.custom_length.setValue(300)
        self.custom_length.setSuffix(" seconds")
        self.custom_length.setEnabled(False)
        length_layout.addWidget(self.custom_length)
        length_layout.addStretch()
        output_layout.addLayout(length_layout)
        
        # Output path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Save to:"))
        self.output_path_label = QLabel("Default: same as input")
        path_layout.addWidget(self.output_path_label)
        output_browse = QPushButton("Choose...")
        output_browse.clicked.connect(self.browse_output)
        path_layout.addWidget(output_browse)
        output_layout.addLayout(path_layout)
        
        # Add music option
        music_layout = QHBoxLayout()
        self.add_music_check = QCheckBox("Add background music")
        music_layout.addWidget(self.add_music_check)
        self.music_path_label = QLabel("No music selected")
        music_layout.addWidget(self.music_path_label)
        music_browse = QPushButton("Browse...")
        music_browse.clicked.connect(self.browse_music)
        music_layout.addWidget(music_browse)
        output_layout.addLayout(music_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Quick settings
        quick_group = QGroupBox("Quick Settings")
        quick_layout = QVBoxLayout()
        
        # Motion sensitivity
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Motion Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(1)
        sens_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label = QLabel("Medium (5)")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{'Low' if v <= 3 else 'High' if v >= 8 else 'Medium'} ({v})")
        )
        sens_layout.addWidget(self.sensitivity_label)
        quick_layout.addLayout(sens_layout)
        
        # Speed smoothing
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Speed Smoothing:"))
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(0, 10)
        self.smoothing_slider.setValue(5)
        smooth_layout.addWidget(self.smoothing_slider)
        self.smoothing_label = QLabel("5")
        self.smoothing_slider.valueChanged.connect(
            lambda v: self.smoothing_label.setText(str(v))
        )
        smooth_layout.addWidget(self.smoothing_label)
        quick_layout.addLayout(smooth_layout)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        # Process button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.process_btn.setMinimumHeight(50)
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready to process")
        progress_layout.addWidget(self.progress_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        return tab
    
    def create_settings_tab(self):
        """Advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Motion detection settings
        motion_group = QGroupBox("Motion Detection Settings")
        motion_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Minimum Motion Duration:"))
        self.min_motion_duration = QDoubleSpinBox()
        self.min_motion_duration.setRange(0.1, 10.0)
        self.min_motion_duration.setValue(0.5)
        self.min_motion_duration.setSuffix(" seconds")
        self.min_motion_duration.setSingleStep(0.1)
        row1.addWidget(self.min_motion_duration)
        row1.addStretch()
        motion_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Motion Threshold:"))
        self.motion_threshold = QSpinBox()
        self.motion_threshold.setRange(1, 100)
        self.motion_threshold.setValue(25)
        row2.addWidget(self.motion_threshold)
        row2.addStretch()
        motion_layout.addLayout(row2)
        
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Blur Size (noise reduction):"))
        self.blur_size = QSpinBox()
        self.blur_size.setRange(3, 51)
        self.blur_size.setValue(21)
        self.blur_size.setSingleStep(2)
        row3.addWidget(self.blur_size)
        row3.addStretch()
        motion_layout.addLayout(row3)
        
        motion_group.setLayout(motion_layout)
        layout.addWidget(motion_group)
        
        # Processing settings
        proc_group = QGroupBox("Processing Settings")
        proc_layout = QVBoxLayout()
        
        self.use_gpu = QCheckBox("Use GPU acceleration (if available)")
        self.use_gpu.setChecked(True)
        proc_layout.addWidget(self.use_gpu)
        
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("CPU Threads:"))
        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, os.cpu_count())
        self.cpu_threads.setValue(max(1, os.cpu_count() - 1))
        thread_layout.addWidget(self.cpu_threads)
        thread_layout.addWidget(QLabel(f"(Max: {os.cpu_count()})"))
        thread_layout.addStretch()
        proc_layout.addLayout(thread_layout)
        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Output Quality:"))
        self.output_quality = QComboBox()
        self.output_quality.addItems(["Low (faster)", "Medium", "High", "Maximum (slower)"])
        self.output_quality.setCurrentIndex(2)
        quality_layout.addWidget(self.output_quality)
        quality_layout.addStretch()
        proc_layout.addLayout(quality_layout)
        
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)
        
        # Save/Load settings
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Settings")
        load_btn.clicked.connect(self.load_settings)
        btn_layout.addWidget(load_btn)
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        btn_layout.addWidget(reset_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        return tab
    
    def create_upload_tab(self):
        """YouTube upload tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("Upload your processed video to YouTube")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Video details
        details_group = QGroupBox("Video Details")
        details_layout = QVBoxLayout()
        
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.yt_title = QLineEdit()
        self.yt_title.setPlaceholderText("Bird Box Activity - [Date]")
        title_layout.addWidget(self.yt_title)
        details_layout.addLayout(title_layout)
        
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.yt_description = QTextEdit()
        self.yt_description.setPlaceholderText("Time-lapse of bird box activity...")
        self.yt_description.setMaximumHeight(100)
        desc_layout.addWidget(self.yt_description)
        details_layout.addLayout(desc_layout)
        
        privacy_layout = QHBoxLayout()
        privacy_layout.addWidget(QLabel("Privacy:"))
        self.yt_privacy = QComboBox()
        self.yt_privacy.addItems(["Private", "Unlisted", "Public"])
        privacy_layout.addWidget(self.yt_privacy)
        privacy_layout.addStretch()
        details_layout.addLayout(privacy_layout)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Upload controls
        upload_btn = QPushButton("Upload to YouTube")
        upload_btn.clicked.connect(self.upload_to_youtube)
        layout.addWidget(upload_btn)
        
        self.upload_progress = QProgressBar()
        layout.addWidget(self.upload_progress)
        
        layout.addStretch()
        
        note = QLabel("Note: First-time upload requires YouTube API authentication")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(note)
        
        return tab
    
    def create_updates_tab(self):
        """Updates checking tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_group = QGroupBox("Application Information")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("Version: 1.0.0"))
        info_layout.addWidget(QLabel("GitHub: [Repository URL]"))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        update_group = QGroupBox("Check for Updates")
        update_layout = QVBoxLayout()
        
        self.update_status = QLabel("Click 'Check for Updates' to check")
        update_layout.addWidget(self.update_status)
        
        check_btn = QPushButton("Check for Updates")
        check_btn.clicked.connect(self.check_updates)
        update_layout.addWidget(check_btn)
        
        update_group.setLayout(update_layout)
        layout.addWidget(update_group)
        
        layout.addStretch()
        return tab
    
    def browse_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv);;All Files (*)"
        )
        if file_path:
            self.input_path_label.setText(file_path)
            self.log_message(f"Selected input: {file_path}")
    
    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Location", "",
            "MP4 Video (*.mp4);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith('.mp4'):
                file_path += '.mp4'
            self.output_path_label.setText(file_path)
            self.log_message(f"Output will be saved to: {file_path}")
    
    def browse_music(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Music File", "",
            "Audio Files (*.mp3 *.wav *.aac *.m4a);;All Files (*)"
        )
        if file_path:
            self.music_path_label.setText(file_path)
            self.add_music_check.setChecked(True)
            self.log_message(f"Selected music: {file_path}")
    
    def on_target_length_changed(self, index):
        self.custom_length.setEnabled(index == 3)  # Enable for "Custom"
    
    def start_processing(self):
        # Validate input
        input_path = self.input_path_label.text()
        if input_path == "No file selected" or not os.path.exists(input_path):
            QMessageBox.warning(self, "Error", "Please select a valid input video file")
            return
        
        # Determine output path
        output_path = self.output_path_label.text()
        if output_path == "Default: same as input":
            output_dir = os.path.dirname(input_path)
            output_filename = Path(input_path).stem + "_timelapse.mp4"
            output_path = os.path.join(output_dir, output_filename)
        
        # Get target length in seconds
        target_idx = self.target_length.currentIndex()
        if target_idx == 0:  # 60 seconds
            target_seconds = 60
        elif target_idx == 1:  # 10 minutes
            target_seconds = 600
        elif target_idx == 2:  # 1 hour
            target_seconds = 3600
        else:  # Custom
            target_seconds = self.custom_length.value()
        
        # Build config
        config = {
            'input_path': input_path,
            'output_path': output_path,
            'target_length': target_seconds,
            'sensitivity': self.sensitivity_slider.value(),
            'smoothing': self.smoothing_slider.value(),
            'min_motion_duration': self.min_motion_duration.value(),
            'motion_threshold': self.motion_threshold.value(),
            'blur_size': self.blur_size.value(),
            'use_gpu': self.use_gpu.isChecked(),
            'cpu_threads': self.cpu_threads.value(),
            'quality': self.output_quality.currentIndex(),
            'add_music': self.add_music_check.isChecked(),
            'music_path': self.music_path_label.text() if self.add_music_check.isChecked() else None
        }
        
        # Disable process button
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_message("Starting processing...")
        
        # Start processing thread
        self.processing_thread = ProcessingThread(config, input_path, output_path)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.start()
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.log_message(message)
    
    def processing_finished(self, success, message):
        self.process_btn.setEnabled(True)
        self.progress_bar.setValue(100 if success else 0)
        self.log_message(message)
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Processing Failed", message)
    
    def log_message(self, message):
        self.log_output.append(f"[{logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}] {message}")
        logger.info(message)
    
    def save_settings(self):
        settings = {
            'motion_sensitivity': self.sensitivity_slider.value(),
            'smoothing': self.smoothing_slider.value(),
            'min_motion_duration': self.min_motion_duration.value(),
            'motion_threshold': self.motion_threshold.value(),
            'blur_size': self.blur_size.value(),
            'use_gpu': self.use_gpu.isChecked(),
            'cpu_threads': self.cpu_threads.value(),
            'output_quality': self.output_quality.currentIndex()
        }
        self.config_manager.save_settings(settings)
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully")
    
    def load_settings(self):
        settings = self.config_manager.load_settings()
        if settings:
            self.sensitivity_slider.setValue(settings.get('motion_sensitivity', 5))
            self.smoothing_slider.setValue(settings.get('smoothing', 5))
            self.min_motion_duration.setValue(settings.get('min_motion_duration', 0.5))
            self.motion_threshold.setValue(settings.get('motion_threshold', 25))
            self.blur_size.setValue(settings.get('blur_size', 21))
            self.use_gpu.setChecked(settings.get('use_gpu', True))
            self.cpu_threads.setValue(settings.get('cpu_threads', max(1, os.cpu_count() - 1)))
            self.output_quality.setCurrentIndex(settings.get('output_quality', 2))
            QMessageBox.information(self, "Settings Loaded", "Settings loaded successfully")
    
    def reset_settings(self):
        self.sensitivity_slider.setValue(5)
        self.smoothing_slider.setValue(5)
        self.min_motion_duration.setValue(0.5)
        self.motion_threshold.setValue(25)
        self.blur_size.setValue(21)
        self.use_gpu.setChecked(True)
        self.cpu_threads.setValue(max(1, os.cpu_count() - 1))
        self.output_quality.setCurrentIndex(2)
        QMessageBox.information(self, "Reset", "Settings reset to defaults")
    
    def upload_to_youtube(self):
        QMessageBox.information(
            self, "YouTube Upload",
            "YouTube upload feature requires OAuth authentication.\n"
            "Please configure your credentials first.\n"
            "See README for setup instructions."
        )
    
    def check_updates(self):
        self.update_status.setText("Checking for updates...")
        try:
            checker = UpdateChecker()
            has_update, version, url = checker.check_for_updates()
            
            if has_update:
                self.update_status.setText(f"New version available: {version}")
                reply = QMessageBox.question(
                    self, "Update Available",
                    f"Version {version} is available. Would you like to download it?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    import webbrowser
                    webbrowser.open(url)
            else:
                self.update_status.setText("You have the latest version")
                QMessageBox.information(self, "Up to Date", "You are running the latest version")
        except Exception as e:
            self.update_status.setText(f"Error checking updates: {str(e)}")
            logger.exception("Error checking for updates")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
