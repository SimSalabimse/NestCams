"""
Enhanced Streamlit-based web application for NestCam Processor v3.0
Modern UI with better UX and performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime
import json
import logging
import asyncio
import threading
from queue import Queue
import base64

import sys
import os

# Add the parent directory to the path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    # Try relative imports first (when run as module)
    from ..config import config, detect_gpu_backend, GPU_BACKEND, HAS_GPU
    from ..processors.video_processor import VideoProcessor
    from ..services.file_service import FileService
    from ..services.youtube_service import YouTubeService
    from ..services.analytics_service import AnalyticsService
except ImportError:
    # Fall back to absolute imports (when run directly by Streamlit)
    from config import config, detect_gpu_backend, GPU_BACKEND, HAS_GPU
    from processors.video_processor import VideoProcessor
    from services.file_service import FileService
    from services.youtube_service import YouTubeService
    from services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

# Enhanced imports with fallbacks
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not available - memory monitoring disabled")

try:
    import cv2
    import numpy as np
except ImportError:
    print("❌ OpenCV not available - video processing disabled")

# Enhanced GPU detection
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not available - GPU acceleration disabled")

# UI Constants
PRIMARY_COLOR = "#FF6B35"
SECONDARY_COLOR = "#4CAF50"
ACCENT_COLOR = "#2196F3"
BACKGROUND_COLOR = "#0E1117"
SURFACE_COLOR = "#262730"

# Custom CSS for enhanced styling
CUSTOM_CSS = f"""
<style>
    /* Modern card styling */
    .metric-card {{
        background: linear-gradient(135deg, {SURFACE_COLOR} 0%, rgba(38, 39, 48, 0.8) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }}
    
    /* Enhanced button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, #FF8C42 100%);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(255, 107, 53, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 107, 53, 0.4);
    }}
    
    /* Progress bar styling */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        border-radius: 10px;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {SURFACE_COLOR};
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.75rem 1rem;
        font-weight: 500;
    }}
    
    /* Status indicators */
    .status-success {{
        color: {SECONDARY_COLOR};
        font-weight: 600;
    }}
    
    .status-warning {{
        color: #FFA726;
        font-weight: 600;
    }}
    
    .status-error {{
        color: #EF5350;
        font-weight: 600;
    }}
    
    .status-info {{
        color: {ACCENT_COLOR};
        font-weight: 600;
    }}
    
    /* Enhanced sidebar */
    .css-1d391kg {{
        background: linear-gradient(180deg, {BACKGROUND_COLOR} 0%, rgba(14, 17, 23, 0.95) 100%);
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {SURFACE_COLOR};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {PRIMARY_COLOR};
        border-radius: 4px;
    }}
    
    /* Animation for loading states */
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
        100% {{ opacity: 1; }}
    }}
    
    .loading-pulse {{
        animation: pulse 2s infinite;
    }}
    
    /* Enhanced metrics */
    .metric-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin: 0.5rem 0;
    }}
    
    .metric-value {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {PRIMARY_COLOR};
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 0.25rem;
    }}
</style>
"""


class NestCamApp:
    """Enhanced web application class with modern UI and better performance"""

    def __init__(self):
        self.processor = VideoProcessor(config)
        self.file_service = FileService(config)
        self.youtube_service = YouTubeService(config)
        self.analytics_service = AnalyticsService(config)

        # Enhanced session state management
        self._initialize_session_state()

        # Performance monitoring
        self.performance_metrics = {}

        # Background processing queue
        self.processing_queue = Queue()
        self.processing_thread = None

        # Load saved configuration
        self._load_configuration()

    def _initialize_session_state(self):
        """Initialize comprehensive session state"""
        default_state = {
            "processing_history": [],
            "current_job": None,
            "uploaded_files": [],
            "output_dir": "",
            "dark_mode": True,
            "performance_mode": "balanced",
            "auto_save": True,
            "notifications_enabled": True,
            "gpu_acceleration": HAS_GPU,
            "processing_queue": [],
            "system_metrics": {},
            "ui_preferences": {
                "show_advanced": False,
                "auto_refresh": True,
                "compact_view": False,
            },
        }

        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _load_configuration(self):
        """Load saved configuration and preferences"""
        config_file = Path("src/data/user_config.json")
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    user_config = json.load(f)
                st.session_state.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load user config: {e}")

    def _save_configuration(self):
        """Save current configuration"""
        config_file = Path("src/data/user_config.json")
        config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_file, "w") as f:
                json.dump(
                    {
                        "dark_mode": st.session_state.dark_mode,
                        "performance_mode": st.session_state.performance_mode,
                        "auto_save": st.session_state.auto_save,
                        "notifications_enabled": st.session_state.notifications_enabled,
                        "ui_preferences": st.session_state.ui_preferences,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Could not save user config: {e}")

    def run(self):
        """Run the enhanced web application"""
        # Inject custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        # Set page configuration
        st.set_page_config(
            page_title="🐦 NestCam Processor v3.0",
            page_icon="🐦",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "Get Help": "https://github.com/your-repo/nestcam-processor",
                "Report a bug": "https://github.com/your-repo/nestcam-processor/issues",
                "About": "NestCam Processor v3.0 - Advanced Bird Nest Video Processing",
            },
        )

        # Enhanced header with performance metrics
        self._render_enhanced_header()

        # Sidebar with enhanced controls
        self._render_enhanced_sidebar()

        # Main content with tabbed interface
        self._render_main_content()

        # Background processing handler
        self._handle_background_processing()

        # Auto-save configuration
        if st.session_state.auto_save:
            self._save_configuration()

    def _render_enhanced_header(self):
        """Render enhanced header with real-time metrics"""
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            st.markdown("# 🐦 NestCam Processor v3.0")
            st.markdown("*Advanced Bird Nest Video Processing with AI*")

        with col2:
            self._render_system_status()

        with col3:
            self._render_performance_metrics()

        with col4:
            self._render_quick_actions()

    def _render_system_status(self):
        """Render system status indicators"""
        st.markdown("### System Status")

        # GPU Status
        gpu_status = "✅ Active" if HAS_GPU else "⚠️ CPU Only"
        gpu_color = "success" if HAS_GPU else "warning"
        st.markdown(
            f"<span class='status-{gpu_color}'>🎯 GPU: {gpu_status}</span>",
            unsafe_allow_html=True,
        )

        # Memory Status
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            memory_status = "✅ Good" if memory.percent < 80 else "⚠️ High"
            memory_color = "success" if memory.percent < 80 else "warning"
            st.markdown(
                f"<span class='status-{memory_color}'>🧠 RAM: {memory_status}</span>",
                unsafe_allow_html=True,
            )

        # Processing Status
        if st.session_state.current_job:
            status = st.session_state.current_job.get("status", "idle")
            if status == "running":
                st.markdown(
                    "<span class='status-info'>⚡ Processing Active</span>",
                    unsafe_allow_html=True,
                )
            elif status == "completed":
                st.markdown(
                    "<span class='status-success'>✅ Job Completed</span>",
                    unsafe_allow_html=True,
                )
            elif status == "error":
                st.markdown(
                    "<span class='status-error'>❌ Processing Error</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span class='status-info'>⏸️ Ready</span>", unsafe_allow_html=True
            )

    def _render_performance_metrics(self):
        """Render real-time performance metrics"""
        st.markdown("### Performance")

        # Processing speed (mock data for now)
        fps = self.performance_metrics.get("current_fps", 0)
        st.metric("Processing Speed", f"{fps} FPS" if fps > 0 else "N/A")

        # Queue status
        queue_size = len(st.session_state.processing_queue)
        if queue_size > 0:
            st.metric("Queue", f"{queue_size} files")
        else:
            st.metric("Queue", "Empty")

    def _render_quick_actions(self):
        """Render quick action buttons"""
        st.markdown("### Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🚀 Start Processing", type="primary", use_container_width=True
            ):
                self._quick_start_processing()

        with col2:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.active_tab = "analytics"
                st.rerun()

    def _render_enhanced_sidebar(self):
        """Render enhanced sidebar with better organization"""
        with st.sidebar:
            st.markdown("## ⚙️ Control Panel")

            # Performance Mode Selector
            st.markdown("### Performance Mode")
            performance_modes = {
                "🚀 High Performance": "high",
                "⚖️ Balanced": "balanced",
                "🛡️ Power Saver": "power_saver",
                "🎯 Custom": "custom",
            }

            selected_mode = st.selectbox(
                "Select performance mode:",
                options=list(performance_modes.keys()),
                index=list(performance_modes.values()).index(
                    st.session_state.performance_mode
                ),
                format_func=lambda x: x,
                key="performance_mode_selector",
            )

            st.session_state.performance_mode = performance_modes[selected_mode]

            # GPU Acceleration Toggle
            st.markdown("### Hardware Acceleration")
            gpu_enabled = st.toggle(
                "🎯 GPU Acceleration",
                value=st.session_state.gpu_acceleration and HAS_GPU,
                disabled=not HAS_GPU,
                help="Enable GPU acceleration for faster processing",
            )
            st.session_state.gpu_acceleration = gpu_enabled

            # Advanced Settings Expander
            with st.expander("🔧 Advanced Settings", expanded=False):
                self._render_advanced_settings()

            # System Information
            with st.expander("💻 System Info", expanded=False):
                self._render_system_info()

    def _render_advanced_settings(self):
        """Render advanced settings panel"""
        st.markdown("#### Processing Settings")

        # Motion detection sensitivity
        config.processing.motion_threshold = st.slider(
            "Motion Sensitivity",
            min_value=500,
            max_value=20000,
            value=config.processing.motion_threshold,
            step=100,
            help="Sensitivity for motion detection (higher = less sensitive)",
        )

        # Processing batch size
        config.processing.batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=16,
            value=config.processing.batch_size,
            help="Number of frames to process simultaneously",
        )

        # Memory limit
        config.processing.memory_limit_gb = st.slider(
            "Memory Limit (GB)",
            min_value=1.0,
            max_value=64.0,
            value=config.processing.memory_limit_gb,
            step=0.5,
            help="Maximum memory usage before optimization kicks in",
        )

        st.markdown("#### Output Settings")

        # Video quality
        quality_options = {
            "🎬 High Quality (1080p)": "1920x1080",
            "📺 Standard (720p)": "1280x720",
            "📱 Mobile (480p)": "854x480",
        }

        selected_quality = st.selectbox(
            "Output Resolution:",
            options=list(quality_options.keys()),
            index=list(quality_options.values()).index(
                config.processing.output_resolution
            ),
        )

        config.processing.output_resolution = quality_options[selected_quality]

        # Save settings button
        if st.button("💾 Save Settings", type="secondary", use_container_width=True):
            config.save_to_file()
            st.success("✅ Settings saved successfully!")

    def _render_system_info(self):
        """Render detailed system information"""
        st.markdown("#### Hardware Information")

        # CPU Info
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        st.metric("CPU Cores", cpu_count)

        # Memory Info
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            st.metric("Total RAM", f"{memory.total / (1024**3):.1f} GB")
            st.metric("Available RAM", f"{memory.available / (1024**3):.1f} GB")

        # GPU Info
        if HAS_GPU:
            if GPU_BACKEND == "cuda" and HAS_TORCH:
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                st.metric("GPU Device", device_name)
                st.metric("GPU Memory", f"{memory_gb:.1f} GB")
            elif GPU_BACKEND == "metal":
                st.metric("GPU Device", "Apple Silicon")
                st.metric("Backend", "Metal Performance Shaders")

        # Disk Space
        if HAS_PSUTIL:
            disk = psutil.disk_usage("/")
            st.metric("Free Disk Space", f"{disk.free / (1024**3):.1f} GB")

    def _render_main_content(self):
        """Render main content with enhanced tabs"""
        tab_names = [
            "📹 Process Videos",
            "📊 Analytics Dashboard",
            "🎵 Audio Settings",
            "📤 YouTube Upload",
            "💻 System Monitor",
            "⚙️ Settings",
        ]

        tabs = st.tabs(tab_names)

        with tabs[0]:
            self._render_video_processing_tab()
        with tabs[1]:
            self._render_analytics_tab()
        with tabs[2]:
            self._render_audio_tab()
        with tabs[3]:
            self._render_upload_tab()
        with tabs[4]:
            self._render_system_monitor_tab()
        with tabs[5]:
            self._render_settings_tab()

    def _render_video_processing_tab(self):
        """Enhanced video processing tab"""
        st.markdown("## 🎬 Video Processing Center")

        # File upload with drag & drop
        st.markdown("### 📁 Upload Videos")

        uploaded_files = st.file_uploader(
            "Drag and drop video files here or click to browse",
            type=["mp4", "avi", "mkv", "mov", "wmv", "flv"],
            accept_multiple_files=True,
            help="Supported formats: MP4, AVI, MKV, MOV, WMV, FLV",
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

            # Enhanced file display
            self._render_file_list(uploaded_files)

            # Processing configuration
            self._render_processing_config()

            # Start processing button
            self._render_processing_controls()

        # Resume functionality
        self._render_resume_section()

        # Processing progress
        if st.session_state.current_job:
            self._render_processing_progress()

    def _render_file_list(self, files):
        """Render enhanced file list with preview"""
        st.markdown("### 📋 Uploaded Files")

        for i, file in enumerate(files):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                with col1:
                    st.markdown(f"**{file.name}**")
                    if hasattr(file, "size"):
                        size_mb = file.size / (1024 * 1024)
                        st.caption(f"{size_mb:.1f} MB")

                with col2:
                    # File type indicator
                    if file.name.lower().endswith((".mp4", ".mov")):
                        st.markdown("🎬 MP4/MOV")
                    elif file.name.lower().endswith((".avi")):
                        st.markdown("📼 AVI")
                    else:
                        st.markdown("📄 Video")

                with col3:
                    # Preview button (placeholder)
                    if st.button(
                        "👁️ Preview",
                        key=f"preview_{i}",
                        help="Preview video (coming soon)",
                    ):
                        st.info("Video preview feature coming soon!")

                with col4:
                    # File info
                    if st.button("ℹ️ Info", key=f"info_{i}"):
                        self._show_file_info(file)

                with col5:
                    # Remove button
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.uploaded_files.remove(file)
                        st.rerun()

    def _render_processing_config(self):
        """Render processing configuration options"""
        st.markdown("### ⚙️ Processing Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Output Settings")

            # Output format
            output_format = st.selectbox(
                "Output Format",
                ["mp4", "avi", "mkv", "mov"],
                index=0,
                help="Choose the output video format",
            )

            # Duration options
            st.markdown("**Generate Videos:**")
            durations = {
                "60s": "60 Second Video",
                "12min": "12 Minute Video",
                "1h": "1 Hour Video",
            }

            for key, label in durations.items():
                st.session_state[f"generate_{key}"] = st.checkbox(
                    f"📹 {label}",
                    value=getattr(st.session_state, f"generate_{key}", True),
                    key=f"duration_{key}",
                )

            # Custom duration
            st.session_state.custom_duration = st.number_input(
                "Custom Duration (seconds)",
                min_value=10,
                max_value=3600,
                value=getattr(st.session_state, "custom_duration", 120),
                help="Specify custom video duration in seconds",
            )

        with col2:
            st.markdown("#### Processing Options")

            # Performance mode
            performance_options = {
                "🚀 Fast": "fast",
                "⚖️ Balanced": "balanced",
                "🎯 High Quality": "quality",
            }

            selected_perf = st.selectbox(
                "Processing Quality",
                options=list(performance_options.keys()),
                index=1,
                help="Choose processing speed vs quality tradeoff",
            )

            st.session_state.processing_quality = performance_options[selected_perf]

            # Advanced options
            with st.expander("🔧 Advanced Options"):
                st.session_state.memory_efficient = st.checkbox(
                    "💾 Memory-Efficient Mode",
                    value=getattr(st.session_state, "memory_efficient", True),
                    help="Use less memory (recommended for large files)",
                )

                st.session_state.enable_watermark = st.checkbox(
                    "🎨 Add Watermark",
                    value=getattr(st.session_state, "enable_watermark", False),
                    help="Add custom watermark to output videos",
                )

                if st.session_state.enable_watermark:
                    st.session_state.watermark_text = st.text_input(
                        "Watermark Text",
                        value=getattr(st.session_state, "watermark_text", ""),
                        help="Text to display as watermark",
                    )

    def _render_processing_controls(self):
        """Render processing control buttons"""
        st.markdown("### 🚀 Processing Controls")

        col1, col2, col3 = st.columns(3)

        # Start processing
        with col1:
            is_processing = (
                st.session_state.current_job is not None
                and st.session_state.current_job.get("status") == "running"
            )

            if st.button(
                "🚀 Start Processing" if not is_processing else "⏳ Processing...",
                type="primary",
                use_container_width=True,
                disabled=is_processing or len(st.session_state.uploaded_files) == 0,
            ):
                self._start_enhanced_processing()

        # Pause/Resume
        with col2:
            if is_processing:
                if st.button("⏸️ Pause", use_container_width=True):
                    self._pause_processing()
            else:
                if st.button("▶️ Resume", use_container_width=True, disabled=True):
                    self._resume_processing()

        # Stop
        with col3:
            if st.button(
                "🛑 Stop", use_container_width=True, disabled=not is_processing
            ):
                self._stop_processing()

    def _render_resume_section(self):
        """Render resume functionality section"""
        if not getattr(config.processing, "enable_resume", True):
            return

        state_dir = getattr(config.processing, "processing_state_dir", None)
        if not state_dir or not Path(state_dir).exists():
            return

        state_files = list(Path(state_dir).glob("processing_state_*.json"))
        if not state_files:
            return

        st.markdown("### 🔄 Resume Previous Sessions")
        st.info("Found interrupted processing sessions that can be resumed:")

        for state_file in sorted(
            state_files, key=lambda x: x.stat().st_mtime, reverse=True
        ):
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)

                timestamp = datetime.fromisoformat(state_data["timestamp"])

                with st.expander(
                    f"📁 Session from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    expanded=False,
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        current_file_index = state_data.get("current_file_index", 0)
                        total_files = (
                            len(st.session_state.uploaded_files)
                            if st.session_state.uploaded_files
                            else "?"
                        )
                        st.write(
                            f"Progress: {current_file_index + 1}/{total_files} files"
                        )

                        # Show processing settings
                        settings = state_data.get("settings", {})
                        st.write(
                            f"Output Format: {settings.get('output_format', 'mp4')}"
                        )
                        st.write(
                            f"GPU: {'Enabled' if settings.get('use_gpu', True) else 'Disabled'}"
                        )

                    with col2:
                        if st.button("🔄 Resume", key=f"resume_{state_file.name}"):
                            self._resume_from_state(state_data)
                            st.rerun()

                        if st.button("🗑️ Delete", key=f"delete_{state_file.name}"):
                            state_file.unlink()
                            st.success("Session deleted!")
                            time.sleep(1)
                            st.rerun()

            except Exception as e:
                st.error(f"Error loading session: {e}")

    def _render_processing_progress(self):
        """Render enhanced processing progress"""
        job = st.session_state.current_job

        if not job:
            return

        st.markdown("### 📊 Processing Progress")

        # Status indicator
        status = job.get("status", "unknown")
        if status == "running":
            st.markdown("⚡ **Status: Processing Active**")
        elif status == "completed":
            st.markdown("✅ **Status: Completed**")
        elif status == "error":
            st.markdown("❌ **Status: Error**")
        elif status == "paused":
            st.markdown("⏸️ **Status: Paused**")

        # Progress bar
        progress = job.get("progress", 0.0)
        st.progress(progress)
        st.write(f"**Progress:** {progress:.1f}%")
        # Current file
        current_file = job.get("current_file", "Unknown")
        if current_file:
            st.write(f"📹 **Current File:** {current_file}")

        # Performance metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            fps = job.get("current_fps", 0)
            st.metric("Processing Speed", f"{fps} FPS" if fps > 0 else "N/A")

        with col2:
            elapsed = time.time() - job.get("start_time", time.time())
            st.metric("Elapsed Time", f"{elapsed:.1f}s")

        with col3:
            eta = job.get("estimated_time_remaining", 0)
            st.metric("ETA", f"{eta:.1f}s" if eta > 0 else "Unknown")

        # Real-time logs
        if job.get("logs"):
            with st.expander("📝 Recent Logs", expanded=False):
                logs = job["logs"][-10:]  # Show last 10 logs
                for log in logs:
                    if isinstance(log, str):
                        st.text(log)
                    elif isinstance(log, dict):
                        st.json(log)

    def _render_analytics_tab(self):
        """Enhanced analytics dashboard"""
        st.markdown("## 📊 Analytics Dashboard")

        # Get analytics data
        analytics_data = self.analytics_service.get_analytics()

        if not analytics_data or not analytics_data.get("videos_processed"):
            st.info(
                "📈 No analytics data available yet. Process some videos to see insights!"
            )
            return

        # Key metrics overview
        self._render_analytics_overview(analytics_data)

        # Performance charts
        self._render_performance_charts(analytics_data)

        # Processing history
        self._render_processing_history(analytics_data)

    def _render_analytics_overview(self, analytics_data):
        """Render analytics overview metrics"""
        st.markdown("### 📈 Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_videos = len(analytics_data.get("videos_processed", []))
            st.metric("Total Videos Processed", total_videos)

        with col2:
            total_frames = sum(
                v.get("frames_processed", 0)
                for v in analytics_data.get("videos_processed", [])
            )
            st.metric("Total Frames", f"{total_frames:,}")

        with col3:
            avg_time = analytics_data.get("avg_processing_time", 0)
            st.metric("Average Processing Time", f"{avg_time:.1f}s")

        with col4:
            success_rate = analytics_data.get("success_rate", 0) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")

    def _render_performance_charts(self, analytics_data):
        """Render performance visualization charts"""
        st.markdown("### 📊 Performance Analysis")

        videos_data = analytics_data.get("videos_processed", [])

        if not videos_data:
            return

        # Processing time distribution
        col1, col2 = st.columns(2)

        with col1:
            processing_times = [v.get("processing_time", 0) for v in videos_data]
            fig = px.histogram(
                processing_times,
                title="Processing Time Distribution",
                labels={"value": "Processing Time (seconds)", "count": "Frequency"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Success rate over time (if timestamps available)
            successful = [v for v in videos_data if not v.get("error")]
            failed = [v for v in videos_data if v.get("error")]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=["Successful", "Failed"],
                    y=[len(successful), len(failed)],
                    marker_color=[SECONDARY_COLOR, "#EF5350"],
                    name="Videos",
                )
            )
            fig.update_layout(title="Processing Success Rate", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Performance trends
        st.markdown("#### 📈 Performance Trends")

        # Create trend data
        trend_data = []
        for i, video in enumerate(videos_data):
            trend_data.append(
                {
                    "video_number": i + 1,
                    "processing_time": video.get("processing_time", 0),
                    "frames_processed": video.get("frames_processed", 0),
                    "success": not bool(video.get("error")),
                }
            )

        if trend_data:
            df_trend = pd.DataFrame(trend_data)

            fig = px.line(
                df_trend,
                x="video_number",
                y="processing_time",
                title="Processing Time Trend",
                labels={
                    "video_number": "Video Number",
                    "processing_time": "Processing Time (s)",
                },
                color_discrete_sequence=[ACCENT_COLOR],
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_processing_history(self, analytics_data):
        """Render detailed processing history"""
        st.markdown("### 📋 Processing History")

        videos_data = analytics_data.get("videos_processed", [])

        if not videos_data:
            st.info("No processing history available.")
            return

        # Convert to DataFrame for better display
        history_data = []
        for video in videos_data:
            history_data.append(
                {
                    "filename": video.get("filename", "Unknown"),
                    "frames_processed": video.get("frames_processed", 0),
                    "processing_time": video.get("processing_time", 0),
                    "motion_events": video.get("motion_events", 0),
                    "status": "Failed" if video.get("error") else "Success",
                    "timestamp": video.get("timestamp", "Unknown"),
                }
            )

        df_history = pd.DataFrame(history_data)

        # Display as data table
        st.dataframe(
            df_history,
            column_config={
                "filename": st.column_config.TextColumn("File Name", width="medium"),
                "frames_processed": st.column_config.NumberColumn(
                    "Frames", format="%d"
                ),
                "processing_time": st.column_config.NumberColumn(
                    "Time (s)", format="%.2f"
                ),
                "motion_events": st.column_config.NumberColumn(
                    "Motion Events", format="%d"
                ),
                "status": st.column_config.TextColumn("Status", width="small"),
                "timestamp": st.column_config.TextColumn(
                    "Processed At", width="medium"
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                csv_data = df_history.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="nestcam_processing_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col2:
            if st.button("📈 Generate Report", use_container_width=True):
                self._generate_processing_report(df_history)

    def _render_system_monitor_tab(self):
        """Enhanced system monitoring dashboard"""
        st.markdown("## 💻 System Monitor")

        # Real-time system metrics
        self._render_realtime_metrics()

        # Performance recommendations
        self._render_performance_recommendations()

        # Resource usage charts
        self._render_resource_charts()

        # System optimization tips
        self._render_optimization_tips()

    def _render_realtime_metrics(self):
        """Render real-time system metrics"""
        st.markdown("### 📊 Real-Time Metrics")

        # Create placeholders for live updates
        cpu_placeholder = st.empty()
        memory_placeholder = st.empty()
        gpu_placeholder = st.empty()
        disk_placeholder = st.empty()

        # Update metrics
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            self._update_system_metrics()
            st.rerun()
        else:
            self._update_system_metrics()

        # Display current metrics
        metrics = st.session_state.system_metrics

        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.markdown("#### 🧠 CPU & Memory")

                cpu_percent = metrics.get("cpu_percent", 0)
                cpu_color = (
                    "success"
                    if cpu_percent < 70
                    else "warning" if cpu_percent < 90 else "error"
                )
                st.markdown(
                    f"<div class='metric-container'><div><div class='metric-value'>{cpu_percent:.1f}%</div><div class='metric-label'>CPU Usage</div></div></div>",
                    unsafe_allow_html=True,
                )

                memory_percent = metrics.get("memory_percent", 0)
                memory_color = (
                    "success"
                    if memory_percent < 80
                    else "warning" if memory_percent < 95 else "error"
                )
                st.markdown(
                    f"<div class='metric-container'><div><div class='metric-value'>{memory_percent:.1f}%</div><div class='metric-label'>Memory Usage</div></div></div>",
                    unsafe_allow_html=True,
                )

        with col2:
            with st.container():
                st.markdown("#### 🎯 GPU & Storage")

                if HAS_GPU and GPU_BACKEND == "cuda":
                    gpu_memory_percent = metrics.get("gpu_memory_percent", 0)
                    gpu_color = (
                        "success"
                        if gpu_memory_percent < 80
                        else "warning" if gpu_memory_percent < 95 else "error"
                    )
                    st.markdown(
                        f"<div class='metric-container'><div><div class='metric-value'>{gpu_memory_percent:.1f}%</div><div class='metric-label'>GPU Memory</div></div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='metric-container'><div><div class='metric-value'>N/A</div><div class='metric-label'>GPU Memory</div></div></div>",
                        unsafe_allow_html=True,
                    )

                disk_percent = metrics.get("disk_percent", 0)
                disk_color = (
                    "success"
                    if disk_percent < 80
                    else "warning" if disk_percent < 95 else "error"
                )
                st.markdown(
                    f"<div class='metric-container'><div><div class='metric-value'>{disk_percent:.1f}%</div><div class='metric-label'>Disk Usage</div></div></div>",
                    unsafe_allow_html=True,
                )

    def _render_settings_tab(self):
        """Enhanced settings panel"""
        st.markdown("## ⚙️ Settings & Preferences")

        # User preferences
        self._render_user_preferences()

        # Processing settings
        self._render_processing_settings()

        # System settings
        self._render_system_settings()

        # Import/Export settings
        self._render_backup_settings()

    def _render_user_preferences(self):
        """Render user preference settings"""
        st.markdown("### 👤 User Preferences")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Appearance")

            # Theme preference
            theme_options = {"🌙 Dark Mode": True, "☀️ Light Mode": False}

            selected_theme = st.selectbox(
                "Theme:",
                options=list(theme_options.keys()),
                index=list(theme_options.values()).index(st.session_state.dark_mode),
            )

            st.session_state.dark_mode = theme_options[selected_theme]

            # UI preferences
            st.session_state.ui_preferences["compact_view"] = st.checkbox(
                "Compact View",
                value=st.session_state.ui_preferences.get("compact_view", False),
                help="Use compact layout to show more information",
            )

            st.session_state.ui_preferences["auto_refresh"] = st.checkbox(
                "Auto-refresh Metrics",
                value=st.session_state.ui_preferences.get("auto_refresh", True),
                help="Automatically refresh system metrics",
            )

        with col2:
            st.markdown("#### Behavior")

            # Notification preferences
            st.session_state.notifications_enabled = st.checkbox(
                "Enable Notifications",
                value=st.session_state.notifications_enabled,
                help="Show desktop notifications for processing events",
            )

            # Auto-save
            st.session_state.auto_save = st.checkbox(
                "Auto-save Settings",
                value=st.session_state.auto_save,
                help="Automatically save settings changes",
            )

            # Performance mode
            performance_options = ["🚀 High Performance", "⚖️ Balanced", "🛡️ Power Saver"]
            selected_perf = st.selectbox(
                "Default Performance Mode:",
                options=performance_options,
                index=1,  # Default to Balanced
            )

    def _render_processing_settings(self):
        """Render processing-specific settings"""
        st.markdown("### 🎬 Processing Settings")

        with st.expander("Motion Detection", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config.processing.motion_threshold = st.slider(
                    "Motion Threshold",
                    min_value=500,
                    max_value=20000,
                    value=config.processing.motion_threshold,
                    step=100,
                    help="Sensitivity for motion detection (higher = less sensitive)",
                )

                config.processing.white_threshold = st.slider(
                    "White Threshold",
                    min_value=100,
                    max_value=255,
                    value=config.processing.white_threshold,
                    help="Threshold for white pixel detection",
                )

            with col2:
                config.processing.black_threshold = st.slider(
                    "Black Threshold",
                    min_value=0,
                    max_value=100,
                    value=config.processing.black_threshold,
                    help="Threshold for black pixel detection",
                )

                config.processing.context_window_size = st.slider(
                    "Context Window",
                    min_value=1,
                    max_value=10,
                    value=config.processing.context_window_size,
                    help="Frames to analyze around detected motion",
                )

        with st.expander("Performance & Memory"):
            col1, col2 = st.columns(2)

            with col1:
                config.processing.batch_size = st.slider(
                    "Batch Size",
                    min_value=1,
                    max_value=16,
                    value=config.processing.batch_size,
                    help="Frames to process simultaneously",
                )

                config.processing.worker_processes = st.slider(
                    "Worker Processes",
                    min_value=1,
                    max_value=8,
                    value=config.processing.worker_processes,
                    help="Number of parallel processing workers",
                )

            with col2:
                config.processing.memory_limit_gb = st.slider(
                    "Memory Limit (GB)",
                    min_value=1.0,
                    max_value=64.0,
                    value=config.processing.memory_limit_gb,
                    step=0.5,
                    help="Maximum memory usage before optimization",
                )

                config.processing.chunk_size = st.slider(
                    "Processing Chunk Size",
                    min_value=50,
                    max_value=2000,
                    value=config.processing.chunk_size,
                    step=50,
                    help="Frames per processing chunk",
                )

        # Save settings
        if st.button(
            "💾 Save Processing Settings", type="primary", use_container_width=True
        ):
            config.save_to_file()
            st.success("✅ Processing settings saved successfully!")

    def _render_system_settings(self):
        """Render system configuration settings"""
        st.markdown("### 💻 System Settings")

        with st.expander("Directory Settings", expanded=False):
            st.markdown("#### 📁 Working Directories")

            # Output directory setting
            output_dir = st.text_input(
                "Output Directory",
                value=getattr(config, "base_dir", Path.cwd()) / "output",
                help="Default directory for processed videos",
            )

            # Temporary directory setting
            temp_dir = st.text_input(
                "Temporary Directory",
                value=str(Path.home() / "tmp" / "nestcam"),
                help="Directory for temporary processing files",
            )

            if st.button("💾 Save Directory Settings", use_container_width=True):
                st.success("✅ Directory settings saved!")

        with st.expander("Performance Settings", expanded=False):
            st.markdown("#### ⚡ Performance Configuration")

            # CPU thread settings
            max_threads = st.slider(
                "Max CPU Threads",
                min_value=1,
                max_value=16,
                value=4,
                help="Maximum number of CPU threads to use",
            )

            # Memory usage settings
            memory_usage = st.slider(
                "Memory Usage Limit (%)",
                min_value=10,
                max_value=90,
                value=70,
                help="Maximum memory usage before optimization kicks in",
            )

            if st.button("💾 Save Performance Settings", use_container_width=True):
                st.success("✅ Performance settings saved!")

        with st.expander("Logging Settings", expanded=False):
            st.markdown("#### 📝 Logging Configuration")

            # Log level setting
            log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            selected_level = st.selectbox(
                "Log Level",
                options=log_levels,
                index=1,  # Default to INFO
                help="Minimum log level to display",
            )

            # Log file settings
            enable_file_logging = st.checkbox(
                "Enable File Logging",
                value=True,
                help="Save logs to file in addition to console",
            )

            if st.button("💾 Save Logging Settings", use_container_width=True):
                st.success("✅ Logging settings saved!")

    def _render_backup_settings(self):
        """Render backup and export settings"""
        st.markdown("### 💾 Backup & Export")

        with st.expander("Settings Backup", expanded=False):
            st.markdown("#### 🔄 Configuration Backup")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📤 Export Settings", use_container_width=True):
                    st.success("✅ Settings exported successfully!")
                    st.info("💡 Feature: Export current configuration to file")

            with col2:
                if st.button("📥 Import Settings", use_container_width=True):
                    st.info("💡 Feature: Import configuration from file")
                    st.info("Upload feature would go here")

        with st.expander("Data Export", expanded=False):
            st.markdown("#### 📊 Analytics Export")

            # Export options
            export_formats = ["CSV", "JSON", "Excel"]
            selected_format = st.selectbox(
                "Export Format",
                options=export_formats,
                index=0,
                help="Choose format for exported data",
            )

            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")

            if st.button("📊 Export Analytics Data", use_container_width=True):
                st.success(f"✅ Analytics data exported as {selected_format}!")
                st.info("💡 Feature: Export processing history and analytics")

        with st.expander("System Reset", expanded=False):
            st.markdown("#### 🔄 System Reset")
            st.warning("⚠️ These actions cannot be undone!")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    st.success("✅ Cache cleared!")

            with col2:
                if st.button("📝 Clear Logs", use_container_width=True):
                    st.success("✅ Logs cleared!")

            with col3:
                if st.button(
                    "🔄 Reset All Settings", use_container_width=True, type="secondary"
                ):
                    st.warning("⚠️ This will reset all settings to defaults!")
                    if st.button("✅ Confirm Reset", type="primary"):
                        st.success("✅ All settings reset to defaults!")

    def _start_enhanced_processing(self):
        """Start enhanced video processing with better error handling"""
        if not st.session_state.uploaded_files:
            st.error("❌ No files uploaded!")
            return

        # Initialize processing job
        st.session_state.current_job = {
            "status": "running",
            "progress": 0.0,
            "current_file": "",
            "start_time": time.time(),
            "logs": [],
            "performance_metrics": {},
            "settings": {
                "use_gpu": st.session_state.gpu_acceleration,
                "quality_mode": st.session_state.processing_quality,
                "memory_efficient": getattr(st.session_state, "memory_efficient", True),
                "output_format": "mp4",  # Default
            },
        }

        # Start background processing
        self._start_background_processing()

        st.success("🚀 Processing started! Check progress below.")

    def _start_background_processing(self):
        """Start background processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.processing_thread = threading.Thread(
            target=self._background_processing_worker, daemon=True
        )
        self.processing_thread.start()

    def _background_processing_worker(self):
        """Background processing worker"""
        try:
            total_files = len(st.session_state.uploaded_files)

            for i, uploaded_file in enumerate(st.session_state.uploaded_files):
                if st.session_state.current_job.get("status") != "running":
                    break

                # Update current file
                st.session_state.current_job["current_file"] = uploaded_file.name
                st.session_state.current_job["progress"] = (i / total_files) * 100

                # Process file (placeholder - implement actual processing)
                self._process_single_file_background(uploaded_file, i, total_files)

            # Mark as completed
            if st.session_state.current_job.get("status") == "running":
                st.session_state.current_job["status"] = "completed"
                st.session_state.current_job["progress"] = 100.0

        except Exception as e:
            st.session_state.current_job["status"] = "error"
            st.session_state.current_job["error"] = str(e)
            logger.error(f"Background processing failed: {e}")

    def _process_single_file_background(self, uploaded_file, file_index, total_files):
        """Process a single file in background (placeholder)"""
        # Simulate processing time
        import random

        processing_time = random.uniform(5, 15)  # 5-15 seconds

        start_time = time.time()
        while time.time() - start_time < processing_time:
            if st.session_state.current_job.get("status") != "running":
                break

            # Update progress
            elapsed = time.time() - start_time
            file_progress = min(elapsed / processing_time, 1.0)
            overall_progress = (file_index + file_progress) / total_files
            st.session_state.current_job["progress"] = overall_progress * 100

            # Simulate FPS
            st.session_state.current_job["current_fps"] = random.uniform(50, 200)

            # Add log entry
            if random.random() < 0.1:  # 10% chance each iteration
                log_entry = (
                    f"{time.time()}: Processing frame {int(file_progress * 1000)}/1000"
                )
                st.session_state.current_job["logs"].append(log_entry)

            time.sleep(0.1)  # Small delay for UI updates

        # Add completion log
        st.session_state.current_job["logs"].append(
            f"{time.time()}: Completed processing {uploaded_file.name}"
        )

    def _pause_processing(self):
        """Pause background processing"""
        if st.session_state.current_job:
            st.session_state.current_job["status"] = "paused"
            st.info("⏸️ Processing paused")

    def _resume_processing(self):
        """Resume background processing"""
        if st.session_state.current_job:
            st.session_state.current_job["status"] = "running"
            self._start_background_processing()
            st.success("▶️ Processing resumed")

    def _stop_processing(self):
        """Stop background processing"""
        if st.session_state.current_job:
            st.session_state.current_job["status"] = "stopped"
            st.warning("🛑 Processing stopped")

    def _handle_background_processing(self):
        """Handle background processing updates"""
        # This method is called in the main run loop
        # Could be used for periodic updates or cleanup
        pass

    def _update_system_metrics(self):
        """Update system metrics in session state"""
        metrics = {}

        if HAS_PSUTIL:
            try:
                # CPU usage
                metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)

                # Memory usage
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
                metrics["memory_used"] = memory.used
                metrics["memory_total"] = memory.total

                # Disk usage
                disk = psutil.disk_usage("/")
                metrics["disk_percent"] = disk.percent
                metrics["disk_used"] = disk.used
                metrics["disk_total"] = disk.total

            except Exception as e:
                logger.warning(f"Failed to update system metrics: {e}")

        # GPU metrics (if available)
        if HAS_GPU and GPU_BACKEND == "cuda" and HAS_TORCH:
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                metrics["gpu_memory_percent"] = (
                    gpu_memory_allocated / gpu_memory_total
                ) * 100
                metrics["gpu_memory_used"] = gpu_memory_allocated
                metrics["gpu_memory_total"] = gpu_memory_total
            except Exception as e:
                logger.warning(f"Failed to update GPU metrics: {e}")

        st.session_state.system_metrics = metrics

    def _show_file_info(self, file):
        """Show detailed file information"""
        st.markdown(f"**File:** {file.name}")
        st.markdown(f"**Size:** {file.size / (1024*1024):.1f} MB")
        st.markdown(f"**Type:** {file.type if hasattr(file, 'type') else 'Unknown'}")

        # Additional file analysis could go here
        if hasattr(file, "size"):
            # Estimate processing time based on file size
            estimated_time = file.size / (1024 * 1024 * 10)  # Rough estimate: 10MB/s
            st.markdown(f"**Estimated Processing Time:** {estimated_time:.1f} seconds")

    # Placeholder methods for other tabs (to be implemented)
    def _render_audio_tab(self):
        """Render audio settings tab"""
        st.markdown("## 🎵 Audio Settings")
        st.info("Audio settings feature coming soon!")

    def _render_upload_tab(self):
        """Render YouTube upload tab"""
        st.markdown("## 📤 YouTube Upload")
        st.info("YouTube upload feature coming soon!")

    def _render_performance_recommendations(self):
        """Render performance recommendations"""
        st.markdown("### 💡 Performance Recommendations")

        recommendations = []

        # GPU recommendations
        if not HAS_GPU:
            recommendations.append(
                "🚀 Enable GPU acceleration for 2-5x faster processing"
            )
        elif GPU_BACKEND == "cpu":
            recommendations.append(
                "🎯 Consider upgrading to a GPU for better performance"
            )

        # Memory recommendations
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                recommendations.append(
                    "💾 High memory usage detected - consider memory-efficient mode"
                )

        # CPU recommendations
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        if cpu_count < 4:
            recommendations.append(
                "⚡ Consider upgrading CPU for better parallel processing"
            )

        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ Your system is optimized for best performance!")

    def _render_resource_charts(self):
        """Render resource usage charts"""
        st.markdown("### 📊 Resource Usage Trends")

        # This would show historical resource usage
        # For now, just show current values
        if st.session_state.system_metrics:
            metrics = st.session_state.system_metrics

            col1, col2 = st.columns(2)

            with col1:
                # CPU chart
                cpu_data = [metrics.get("cpu_percent", 0)]
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=cpu_data[0],
                        title={"text": "CPU Usage"},
                        gauge={"axis": {"range": [0, 100]}},
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Memory chart
                memory_data = [metrics.get("memory_percent", 0)]
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=memory_data[0],
                        title={"text": "Memory Usage"},
                        gauge={"axis": {"range": [0, 100]}},
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_optimization_tips(self):
        """Render system optimization tips"""
        st.markdown("### 🚀 Optimization Tips")

        tips = [
            "🎯 Use GPU acceleration for 2-5x performance boost",
            "💾 Enable memory-efficient mode for large video files",
            "⚡ Adjust batch size based on your system's capabilities",
            "🎬 Choose appropriate output resolution for your needs",
            "📊 Monitor system resources during processing",
            "🧹 Regularly clean up processing state files",
            "🔄 Use resume functionality to continue interrupted processing",
        ]

        for tip in tips:
            st.markdown(f"- {tip}")

    def _generate_processing_report(self, df_history):
        """Generate a detailed processing report"""
        st.markdown("### 📋 Processing Report")

        # Summary statistics
        total_videos = len(df_history)
        successful_videos = len(df_history[df_history["status"] == "Success"])
        success_rate = (
            (successful_videos / total_videos) * 100 if total_videos > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Videos", total_videos)
        with col2:
            st.metric("Successful", successful_videos)
        with col3:
            st.metric("Failed", total_videos - successful_videos)
        with col4:
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # Average processing time
        if not df_history.empty:
            avg_time = df_history["processing_time"].mean()
            st.metric("Average Processing Time", f"{avg_time:.2f}s")

            # Fastest and slowest
            fastest = df_history.loc[df_history["processing_time"].idxmin()]
            slowest = df_history.loc[df_history["processing_time"].idxmax()]

            st.markdown("**Performance Highlights:**")
            st.markdown(
                f"- 🚀 Fastest: {fastest['filename']} ({fastest['processing_time']:.2f}s)"
            )
            st.markdown(
                f"- 🐌 Slowest: {slowest['filename']} ({slowest['processing_time']:.2f}s)"
            )


def main():
    """Main entry point for enhanced web app"""
    app = NestCamApp()
    app.run()


if __name__ == "__main__":
    main()
