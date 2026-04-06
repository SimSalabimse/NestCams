#!/usr/bin/env python3
"""
Bird Motion Video Processor v2.0
Enhanced with AI detection, real-time monitoring, analytics, and advanced processing
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QProgressBar,
                             QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QCheckBox, QTextEdit, QTabWidget,
                             QMessageBox, QLineEdit, QSlider, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon

from motion_detector import MotionDetector
from video_processor import VideoProcessor
from youtube_uploader import YouTubeUploader
from update_checker import UpdateChecker
from config_manager import ConfigManager
from analytics_dashboard import AnalyticsDashboard
from real_time_monitor import RealTimeMonitor
from bird_detector import get_bird_detector

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
    """Enhanced background thread with cancellation and time estimates"""
    progress_update = pyqtSignal(int, str)
    time_estimate = pyqtSignal(float)  # Seconds remaining
    processing_complete = pyqtSignal(bool, str, dict)  # success, message, stats
    
    def __init__(self, config, video_path, output_paths, batch_mode=False):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.output_paths = output_paths
        self.batch_mode = batch_mode
        self.start_time = None
        self.cancelled = False
        
    def cancel(self):
        """Cancel processing"""
        self.cancelled = True
        logger.info("Cancellation requested")
        
    def run(self):
        try:
            self.start_time = time.time()
            
            # Step 1: Detect motion
            self.progress_update.emit(5, "Analyzing video for motion...")
            detector = MotionDetector(self.config)
            motion_segments, stats = detector.detect_motion(
                self.video_path,
                progress_callback=lambda p: self._update_progress(5, 45, p, "Detecting motion")
            )
            
            if self.cancelled:
                self.processing_complete.emit(False, "Processing cancelled", {})
                return
            
            if not motion_segments:
                self.processing_complete.emit(False, "No motion detected in video", {})
                return
            
            # Step 2: Process video(s)
            self.progress_update.emit(50, "Processing video segments...")
            processor = VideoProcessor(self.config)
            
            # Set cancel handler
            self.cancelled_check = lambda: self.cancelled
            
            if self.batch_mode:
                # Process all lengths
                results = processor.create_timelapse_batch(
                    self.video_path,
                    motion_segments,
                    self.output_paths,
                    progress_callback=lambda p: self._update_progress(50, 100, p, "Creating time-lapses"),
                    time_estimate_callback=self.time_estimate.emit
                )
                
                if results:
                    message = f"Created {len(results)} time-lapses:\n"
                    for length, path in results.items():
                        message += f"  {length}s → {path}\n"
                    
                    stats['output_files'] = results
                    stats['processing_time'] = time.time() - self.start_time
                    
                    self.processing_complete.emit(True, message, stats)
                else:
                    self.processing_complete.emit(False, "Failed to create time-lapses", {})
            else:
                # Single time-lapse
                success = processor.create_timelapse(
                    self.video_path,
                    motion_segments,
                    self.output_paths,
                    progress_callback=lambda p: self._update_progress(50, 100, p, "Creating time-lapse"),
                    time_estimate_callback=self.time_estimate.emit
                )
                
                if success:
                    stats['output_file'] = self.output_paths
                    stats['processing_time'] = time.time() - self.start_time
                    self.processing_complete.emit(True, f"Video saved to: {self.output_paths}", stats)
                else:
                    self.processing_complete.emit(False, "Failed to process video", {})
                
        except Exception as e:
            logger.exception("Error during processing")
            self.processing_complete.emit(False, f"Error: {str(e)}", {})
    
    def _update_progress(self, start, end, percent, message):
        """Update progress with interpolation"""
        if self.cancelled:
            return
        progress = int(start + (end - start) * (percent / 100))
        self.progress_update.emit(progress, f"{message}: {percent:.1f}%")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.analytics = AnalyticsDashboard()
        self.processing_thread = None
        self.real_time_monitor = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Bird Motion Video Processor v2.0")
        self.setGeometry(100, 100, 1000, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("Bird Box Motion Processor v2.0")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_main_tab(), "Process Video")
        tabs.addTab(self.create_settings_tab(), "Settings")
        tabs.addTab(self.create_upload_tab(), "YouTube Upload")
        tabs.addTab(self.create_analytics_tab(), "Analytics")
        tabs.addTab(self.create_monitoring_tab(), "Real-Time Monitor")
        tabs.addTab(self.create_updates_tab(), "Updates")
        layout.addWidget(tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready - Enhanced v2.0")
        
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
        
        # Batch mode checkbox
        self.batch_mode = QCheckBox("Process ALL lengths at once (60s, 10min, 1hr)")
        self.batch_mode.setChecked(False)
        self.batch_mode.stateChanged.connect(self.on_batch_mode_changed)
        output_layout.addWidget(self.batch_mode)
        
        # Target length (for single mode)
        self.length_widget = QWidget()
        length_layout = QHBoxLayout(self.length_widget)
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
        output_layout.addWidget(self.length_widget)
        
        # Output path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Save to:"))
        self.output_path_label = QLabel("Default: same as input")
        path_layout.addWidget(self.output_path_label)
        output_browse = QPushButton("Choose...")
        output_browse.clicked.connect(self.browse_output)
        path_layout.addWidget(output_browse)
        output_layout.addLayout(path_layout)
        
        # Add music options
        music_group = QGroupBox("Background Music")
        music_layout = QVBoxLayout()
        
        self.add_music_check = QCheckBox("Add background music")
        music_layout.addWidget(self.add_music_check)
        
        # Music for different lengths
        self.music_60s_layout = QHBoxLayout()
        self.music_60s_layout.addWidget(QLabel("60s video:"))
        self.music_60s_label = QLabel("No music")
        self.music_60s_layout.addWidget(self.music_60s_label)
        music_60s_btn = QPushButton("Browse...")
        music_60s_btn.clicked.connect(lambda: self.browse_music(60))
        self.music_60s_layout.addWidget(music_60s_btn)
        music_layout.addLayout(self.music_60s_layout)
        
        self.music_10min_layout = QHBoxLayout()
        self.music_10min_layout.addWidget(QLabel("10min video:"))
        self.music_10min_label = QLabel("No music")
        self.music_10min_layout.addWidget(self.music_10min_label)
        music_10min_btn = QPushButton("Browse...")
        music_10min_btn.clicked.connect(lambda: self.browse_music(600))
        self.music_10min_layout.addWidget(music_10min_btn)
        music_layout.addLayout(self.music_10min_layout)
        
        self.music_1hr_layout = QHBoxLayout()
        self.music_1hr_layout.addWidget(QLabel("1hr video:"))
        self.music_1hr_label = QLabel("No music")
        self.music_1hr_layout.addWidget(self.music_1hr_label)
        music_1hr_btn = QPushButton("Browse...")
        music_1hr_btn.clicked.connect(lambda: self.browse_music(3600))
        self.music_1hr_layout.addWidget(music_1hr_btn)
        music_layout.addLayout(self.music_1hr_layout)
        
        music_group.setLayout(music_layout)
        output_layout.addWidget(music_group)
        
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
        
        # Enhancement options
        self.motion_blur_check = QCheckBox("Add motion blur (smoother video)")
        self.motion_blur_check.setChecked(True)
        quick_layout.addWidget(self.motion_blur_check)
        
        self.smooth_transitions_check = QCheckBox("Smooth transitions between segments")
        self.smooth_transitions_check.setChecked(True)
        quick_layout.addWidget(self.smooth_transitions_check)
        
        self.color_correction_check = QCheckBox("Apply color correction")
        self.color_correction_check.setChecked(True)
        quick_layout.addWidget(self.color_correction_check)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        # Process buttons
        button_layout = QHBoxLayout()
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.process_btn.setMinimumHeight(50)
        self.process_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.process_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFont(QFont("Arial", 12))
        self.cancel_btn.setMinimumHeight(50)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready to process")
        progress_layout.addWidget(self.progress_label)
        self.time_estimate_label = QLabel("")
        progress_layout.addWidget(self.time_estimate_label)
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
        
        # Music paths storage
        self.music_paths = {60: None, 600: None, 3600: None}
        
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
        
        # Frame filtering settings
        filter_group = QGroupBox("Frame Quality Filtering")
        filter_layout = QVBoxLayout()
        
        filter_layout.addWidget(QLabel("Automatically remove bad frames:"))
        
        self.filter_white_frames = QCheckBox("White frames (overexposed/pause screens)")
        self.filter_white_frames.setChecked(True)
        filter_layout.addWidget(self.filter_white_frames)
        
        self.filter_black_frames = QCheckBox("Black frames (underexposed/stream down)")
        self.filter_black_frames.setChecked(True)
        filter_layout.addWidget(self.filter_black_frames)
        
        self.filter_corrupted_frames = QCheckBox("Corrupted frames (green/purple artifacts)")
        self.filter_corrupted_frames.setChecked(True)
        filter_layout.addWidget(self.filter_corrupted_frames)
        
        self.filter_blurry_frames = QCheckBox("Blurry frames (out of focus)")
        self.filter_blurry_frames.setChecked(True)
        filter_layout.addWidget(self.filter_blurry_frames)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
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
    
    def create_analytics_tab(self):
        """Analytics dashboard tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Summary statistics
        summary_group = QGroupBox("Processing Statistics")
        summary_layout = QVBoxLayout()
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        summary_layout.addWidget(self.stats_table)
        
        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self.refresh_analytics)
        summary_layout.addWidget(refresh_btn)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Export button
        export_btn = QPushButton("Export Statistics to JSON")
        export_btn.clicked.connect(self.export_analytics)
        layout.addWidget(export_btn)
        
        layout.addStretch()
        
        # Load initial stats
        self.refresh_analytics()
        
        return tab
    
    def create_monitoring_tab(self):
        """Real-time monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("Automatically process new videos in a folder")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Folder selection
        folder_group = QGroupBox("Monitor Folder")
        folder_layout = QHBoxLayout()
        
        self.monitor_folder_label = QLabel("No folder selected")
        folder_layout.addWidget(self.monitor_folder_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_monitor_folder)
        folder_layout.addWidget(browse_btn)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # Monitor controls
        control_layout = QHBoxLayout()
        
        self.start_monitor_btn = QPushButton("Start Monitoring")
        self.start_monitor_btn.clicked.connect(self.start_monitoring)
        control_layout.addWidget(self.start_monitor_btn)
        
        self.stop_monitor_btn = QPushButton("Stop Monitoring")
        self.stop_monitor_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitor_btn.setEnabled(False)
        control_layout.addWidget(self.stop_monitor_btn)
        
        layout.addLayout(control_layout)
        
        # Status
        self.monitor_status_label = QLabel("Status: Not monitoring")
        layout.addWidget(self.monitor_status_label)
        
        layout.addStretch()
        
        return tab
    
    def create_updates_tab(self):
        """Updates checking tab with configurable repo"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_group = QGroupBox("Application Information")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("Version: 2.0.0 (Enhanced)"))
        info_layout.addWidget(QLabel("New Features: AI detection, real-time monitoring, analytics, batch processing"))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # GitHub repository configuration
        repo_group = QGroupBox("GitHub Repository")
        repo_layout = QVBoxLayout()
        
        repo_layout.addWidget(QLabel("Repository (owner/repo):"))
        
        repo_input_layout = QHBoxLayout()
        self.repo_input = QLineEdit()
        checker = UpdateChecker()
        self.repo_input.setText(checker.get_repository())
        repo_input_layout.addWidget(self.repo_input)
        
        save_repo_btn = QPushButton("Save")
        save_repo_btn.clicked.connect(self.save_repository)
        repo_input_layout.addWidget(save_repo_btn)
        
        repo_layout.addLayout(repo_input_layout)
        repo_group.setLayout(repo_layout)
        layout.addWidget(repo_group)
        
        # Update checking
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
    
    def on_batch_mode_changed(self, state):
        """Handle batch mode checkbox"""
        self.length_widget.setEnabled(not state)
    
    def on_target_length_changed(self, index):
        self.custom_length.setEnabled(index == 3)
    
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
    
    def browse_music(self, length):
        """Browse for music file for specific length"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Music for {length}s video", "",
            "Audio Files (*.mp3 *.wav *.aac *.m4a);;All Files (*)"
        )
        if file_path:
            self.music_paths[length] = file_path
            self.add_music_check.setChecked(True)
            
            # Update label
            if length == 60:
                self.music_60s_label.setText(Path(file_path).name)
            elif length == 600:
                self.music_10min_label.setText(Path(file_path).name)
            elif length == 3600:
                self.music_1hr_label.setText(Path(file_path).name)
            
            self.log_message(f"Selected music for {length}s: {file_path}")
    
    def browse_monitor_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Monitor")
        if folder:
            self.monitor_folder_label.setText(folder)
    
    def start_monitoring(self):
        """Start real-time folder monitoring"""
        folder = self.monitor_folder_label.text()
        if folder == "No folder selected":
            QMessageBox.warning(self, "Error", "Please select a folder to monitor")
            return
        
        if not os.path.exists(folder):
            QMessageBox.warning(self, "Error", "Selected folder does not exist")
            return
        
        # Create monitor
        self.real_time_monitor = RealTimeMonitor(folder, self.process_new_video)
        
        if self.real_time_monitor.start():
            self.start_monitor_btn.setEnabled(False)
            self.stop_monitor_btn.setEnabled(True)
            self.monitor_status_label.setText(f"Status: Monitoring {folder}")
            self.log_message(f"Started monitoring: {folder}")
        else:
            QMessageBox.warning(self, "Error", "Failed to start monitoring")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.real_time_monitor:
            self.real_time_monitor.stop()
            self.start_monitor_btn.setEnabled(True)
            self.stop_monitor_btn.setEnabled(False)
            self.monitor_status_label.setText("Status: Not monitoring")
            self.log_message("Stopped monitoring")
    
    def process_new_video(self, video_path):
        """Process newly detected video"""
        self.log_message(f"Auto-processing: {video_path}")
        # Auto-set input and process
        self.input_path_label.setText(video_path)
        self.start_processing()
    
    def start_processing(self):
        """Start video processing"""
        # Validate input
        input_path = self.input_path_label.text()
        if input_path == "No file selected" or not os.path.exists(input_path):
            QMessageBox.warning(self, "Error", "Please select a valid input video file")
            return
        
        # Determine output path(s)
        output_path = self.output_path_label.text()
        if output_path == "Default: same as input":
            output_dir = os.path.dirname(input_path)
            output_filename = Path(input_path).stem + "_timelapse"
            output_path = os.path.join(output_dir, output_filename)
        
        # Build config
        config = self._build_config()
        
        batch_mode = self.batch_mode.isChecked()
        
        # Disable process button, enable cancel
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_message("Starting processing...")
        
        # Start processing thread
        self.processing_thread = ProcessingThread(config, input_path, output_path, batch_mode)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.time_estimate.connect(self.update_time_estimate)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.start()
    
    def cancel_processing(self):
        """Cancel ongoing processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, "Cancel Processing",
                "Are you sure you want to cancel processing?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processing_thread.cancel()
                self.log_message("Cancelling processing...")
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.log_message(message)
    
    def update_time_estimate(self, seconds_remaining):
        """Update time estimate display"""
        if seconds_remaining > 0:
            eta = datetime.now() + timedelta(seconds=seconds_remaining)
            self.time_estimate_label.setText(
                f"Estimated time remaining: {int(seconds_remaining//60)}m {int(seconds_remaining%60)}s "
                f"(ETA: {eta.strftime('%H:%M:%S')})"
            )
        else:
            self.time_estimate_label.setText("")
    
    def processing_finished(self, success, message, stats):
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        self.time_estimate_label.setText("")
        self.log_message(message)
        
        if success:
            # Log to analytics
            self.analytics.log_processing(stats)
            
            # Show success
            QMessageBox.information(self, "Success", message)
            
            # Refresh analytics
            self.refresh_analytics()
        else:
            QMessageBox.warning(self, "Processing Failed", message)
    
    def _build_config(self) -> dict:
        """Build configuration dictionary from UI"""
        # Get target length(s)
        if self.batch_mode.isChecked():
            target_lengths = [60, 600, 3600]
        else:
            target_idx = self.target_length.currentIndex()
            if target_idx == 0:
                target_lengths = [60]
            elif target_idx == 1:
                target_lengths = [600]
            elif target_idx == 2:
                target_lengths = [3600]
            else:
                target_lengths = [self.custom_length.value()]
        
        return {
            'sensitivity': self.sensitivity_slider.value(),
            'min_motion_duration': self.min_motion_duration.value(),
            'motion_threshold': self.motion_threshold.value(),
            'blur_size': self.blur_size.value(),
            'use_gpu': self.use_gpu.isChecked(),
            'cpu_threads': self.cpu_threads.value(),
            'quality': self.output_quality.currentIndex(),
            'add_music': self.add_music_check.isChecked(),
            'music_paths': self.music_paths,
            'target_lengths': target_lengths,
            'motion_blur': self.motion_blur_check.isChecked(),
            'smooth_transitions': self.smooth_transitions_check.isChecked(),
            'color_correction': self.color_correction_check.isChecked(),
            'filter_white': self.filter_white_frames.isChecked(),
            'filter_black': self.filter_black_frames.isChecked(),
            'filter_corrupted': self.filter_corrupted_frames.isChecked(),
            'filter_blurry': self.filter_blurry_frames.isChecked()
        }
    
    def log_message(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_output.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def save_settings(self):
        settings = self._build_config()
        self.config_manager.save_settings(settings)
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully")
    
    def load_settings(self):
        settings = self.config_manager.load_settings()
        if settings:
            self.sensitivity_slider.setValue(settings.get('motion_sensitivity', 5))
            self.min_motion_duration.setValue(settings.get('min_motion_duration', 0.5))
            self.motion_threshold.setValue(settings.get('motion_threshold', 25))
            self.blur_size.setValue(settings.get('blur_size', 21))
            self.use_gpu.setChecked(settings.get('use_gpu', True))
            self.cpu_threads.setValue(settings.get('cpu_threads', max(1, os.cpu_count() - 1)))
            self.output_quality.setCurrentIndex(settings.get('output_quality', 2))
            QMessageBox.information(self, "Settings Loaded", "Settings loaded successfully")
    
    def reset_settings(self):
        self.sensitivity_slider.setValue(5)
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
    
    def refresh_analytics(self):
        """Refresh analytics display"""
        stats = self.analytics.get_summary(30)
        
        self.stats_table.setRowCount(0)
        
        # Add rows
        rows = [
            ("Total Videos Processed", str(stats['total_videos_processed'])),
            ("Videos (Last 30 Days)", str(stats['recent_videos'])),
            ("Motion Time (Last 30 Days)", f"{stats['recent_motion_time_hours']:.1f} hours"),
            ("Avg Motion Percentage", f"{stats['average_motion_percentage']:.1f}%"),
            ("Processing Time (Last 30 Days)", f"{stats['total_processing_time_hours']:.1f} hours"),
            ("White Frames Filtered", str(stats['frames_filtered']['white'])),
            ("Black Frames Filtered", str(stats['frames_filtered']['black'])),
            ("Corrupted Frames Filtered", str(stats['frames_filtered']['corrupted'])),
            ("Blurry Frames Filtered", str(stats['frames_filtered']['blurry'])),
        ]
        
        for i, (metric, value) in enumerate(rows):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))
    
    def export_analytics(self):
        """Export analytics to JSON"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics", "statistics.json",
            "JSON Files (*.json)"
        )
        if file_path:
            self.analytics.export_statistics(file_path)
            QMessageBox.information(self, "Exported", f"Statistics exported to {file_path}")
    
    def save_repository(self):
        """Save GitHub repository URL"""
        repo = self.repo_input.text().strip()
        checker = UpdateChecker()
        
        if checker.set_repository(repo):
            QMessageBox.information(self, "Saved", f"Repository set to: {repo}")
        else:
            QMessageBox.warning(self, "Invalid Format", 
                              "Please use format: owner/repo\n"
                              "Example: yourusername/bird-motion-processor")
    
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
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
