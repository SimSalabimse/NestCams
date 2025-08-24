"""
Streamlit-based web application for NestCam Processor v2.0
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

import sys
import os

# Add the parent directory to the path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    # Try relative imports first (when run as module)
    from ..config import config
    from ..processors.video_processor import VideoProcessor
    from ..services.file_service import FileService
    from ..services.youtube_service import YouTubeService
    from ..services.analytics_service import AnalyticsService
except ImportError:
    # Fall back to absolute imports (when run directly by Streamlit)
    from config import config
    from processors.video_processor import VideoProcessor
    from services.file_service import FileService
    from services.youtube_service import YouTubeService
    from services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

import cv2
import numpy as np

# Optional imports with fallbacks
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not available - memory monitoring disabled")

import gc
from concurrent.futures import ProcessPoolExecutor

try:
    import tempfile
    import shutil
except ImportError:
    print("⚠️ tempfile/shutil not available - some features may not work")

import time

# GPU Support (CUDA for 4070ti)
try:
    import cupy as cp

    HAS_GPU = True
    print("✅ GPU acceleration enabled (CUDA)")
except ImportError:
    HAS_GPU = False
    print("⚠️ GPU acceleration not available (install cupy for CUDA support)")

import re


# Memory monitoring (optional)
def get_memory_usage():
    """Get current memory usage in GB"""
    if HAS_PSUTIL:
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except:
            return 0.0
    else:
        return 0.0  # Return 0 if psutil not available


def log_memory_usage(message):
    """Log current memory usage"""
    mem_gb = get_memory_usage()
    if mem_gb > 0:
        print(f"{message} - Memory: {mem_gb:.2f}GB")
    else:
        print(f"{message} - Memory monitoring not available")


class NestCamApp:
    """Main web application class"""

    def __init__(self):
        self.processor = VideoProcessor(config)
        self.file_service = FileService(config)
        self.youtube_service = YouTubeService(config)
        self.analytics_service = AnalyticsService(config)

        # Initialize session state
        if "processing_history" not in st.session_state:
            st.session_state.processing_history = []
        if "current_job" not in st.session_state:
            st.session_state.current_job = None
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "output_dir" not in st.session_state:
            st.session_state.output_dir = ""
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False

        # Load saved stats
        self._load_stats()

    def _load_stats(self):
        """Load processing statistics from disk"""
        stats_file = config.data_dir / "processing_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    saved_stats = json.load(f)
                    if "processing_history" in saved_stats:
                        st.session_state.processing_history = saved_stats[
                            "processing_history"
                        ]
            except Exception as e:
                logger.warning(f"Could not load saved stats: {e}")

    def _save_stats(self):
        """Save processing statistics to disk"""
        stats_file = config.data_dir / "processing_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(stats_file, "w") as f:
                json.dump(
                    {
                        "processing_history": st.session_state.processing_history,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    default=str,
                )
        except Exception as e:
            logger.warning(f"Could not save stats: {e}")

    def _validate_progress(self, progress):
        """Validate and convert progress value to valid numeric format"""
        try:
            if isinstance(progress, str):
                # Try to extract numeric value from string
                numeric_match = re.search(r"(\d+\.?\d*)", progress)
                if numeric_match:
                    progress = float(numeric_match.group(1))
                else:
                    progress = 0.0
            elif progress is None:
                progress = 0.0
            elif not isinstance(progress, (int, float)):
                progress = 0.0

            # Ensure progress is in valid range (0.0 to 100.0)
            progress = max(0.0, min(100.0, float(progress)))

            return progress
        except Exception as e:
            logger.warning(f"Progress validation error: {e}")
            return 0.0

    def run(self):
        """Run the Streamlit application"""
        # Check dark mode preference from session state
        dark_mode = st.session_state.get("dark_mode", False)

        # Set page config (theme is controlled by .streamlit/config.toml)
        st.set_page_config(
            page_title="NestCam Processor v2.0",
            page_icon="🐦",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🐦 NestCam Processor v2.0")
        st.markdown("*Advanced Bird Nest Video Processing*")

        # Sidebar
        self._render_sidebar()

        # Main content
        self._render_main_content()

    def _render_sidebar(self):
        """Render sidebar with settings and controls"""
        with st.sidebar:
            st.header("⚙️ Settings")

            # Dark Mode Toggle (for future theme switching)
            st.markdown("### 🌙 Theme Settings")
            st.info(
                '🌟 **Pro Tip:** To change between light and dark themes, edit the `.streamlit/config.toml` file in your project root and change `base = "dark"` to `base = "light"`'
            )

            # Store preference for future use
            dark_mode_preferred = st.checkbox(
                "I prefer dark mode",
                value=st.session_state.get("dark_mode_preferred", False),
                help="This setting will be used when theme switching is implemented",
            )
            st.session_state.dark_mode_preferred = dark_mode_preferred

            if dark_mode_preferred:
                st.success("✅ Your preference for dark mode has been saved!")
                st.info(
                    '📝 **To apply:** Edit `.streamlit/config.toml` and set `base = "dark"`'
                )

            st.divider()

            # Video processing settings
            st.subheader("Video Processing")
            config.processing.motion_threshold = st.slider(
                "Motion Threshold",
                min_value=500,
                max_value=20000,
                value=config.processing.motion_threshold,
                step=100,
            )

            config.processing.white_threshold = st.slider(
                "White Threshold",
                min_value=100,
                max_value=255,
                value=config.processing.white_threshold,
            )

            config.processing.black_threshold = st.slider(
                "Black Threshold",
                min_value=0,
                max_value=100,
                value=config.processing.black_threshold,
            )

            st.divider()

            # Detailed Analysis Settings - Store in session state first
            st.subheader("Motion Detection Analysis")

            # Get current values with defaults
            current_use_detailed = getattr(
                config.processing, "use_detailed_analysis", True
            )
            current_detail_level = getattr(config.processing, "detail_level", "normal")
            current_context_window = getattr(
                config.processing, "context_window_size", 3
            )
            current_methods = getattr(
                config.processing,
                "analysis_methods",
                ["white_threshold", "motion_diff"],
            )

            # Store in session state to avoid config assignment issues
            if "detailed_analysis_settings" not in st.session_state:
                st.session_state.detailed_analysis_settings = {
                    "use_detailed_analysis": current_use_detailed,
                    "detail_level": current_detail_level,
                    "context_window_size": current_context_window,
                    "analysis_methods": current_methods,
                }

            # Use session state values
            use_detailed = st.checkbox(
                "Use Detailed Analysis",
                value=st.session_state.detailed_analysis_settings[
                    "use_detailed_analysis"
                ],
                help="Enable Pass 2 detailed motion analysis (slower but more accurate)",
            )
            st.session_state.detailed_analysis_settings["use_detailed_analysis"] = (
                use_detailed
            )

            if use_detailed:
                # Detail level selection
                detail_options = {
                    "light": "⚡ Light - Fast, basic analysis",
                    "normal": "🔍 Normal - Balanced speed/accuracy",
                    "detailed": "🎯 Detailed - Slow, comprehensive analysis",
                }

                selected_detail = st.selectbox(
                    "Analysis Detail Level",
                    options=list(detail_options.keys()),
                    format_func=lambda x: detail_options[x],
                    index=list(detail_options.keys()).index(
                        st.session_state.detailed_analysis_settings["detail_level"]
                    ),
                    help="Higher detail = better accuracy but slower processing",
                )
                st.session_state.detailed_analysis_settings["detail_level"] = (
                    selected_detail
                )

                # Context window size
                context_window = st.slider(
                    "Context Window Size",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.detailed_analysis_settings[
                        "context_window_size"
                    ],
                    help="Frames around detected motion to analyze (higher = more accurate but slower)",
                )
                st.session_state.detailed_analysis_settings["context_window_size"] = (
                    context_window
                )

                # Analysis methods
                st.markdown("**Analysis Methods**")
                available_methods = {
                    "white_threshold": "White threshold detection",
                    "black_threshold": "Black threshold detection",
                    "motion_diff": "Motion difference analysis",
                    "edge_detection": "Edge detection",
                    "histogram": "Color histogram comparison",
                }

                selected_methods = []
                for method_key, method_desc in available_methods.items():
                    if st.checkbox(
                        method_desc,
                        value=method_key
                        in st.session_state.detailed_analysis_settings[
                            "analysis_methods"
                        ],
                        key=f"method_{method_key}",
                    ):
                        selected_methods.append(method_key)

                st.session_state.detailed_analysis_settings["analysis_methods"] = (
                    selected_methods
                )

                # Performance info
                if selected_detail == "light":
                    st.info(
                        "⚡ **Light Mode**: ~2-3x faster, good for basic motion detection"
                    )
                elif selected_detail == "normal":
                    st.info(
                        "🔍 **Normal Mode**: Balanced performance, recommended for most uses"
                    )
                else:  # detailed
                    st.warning(
                        "🎯 **Detailed Mode**: ~5-10x slower, best accuracy for critical analysis"
                    )
            else:
                st.info("⚡ **Fast Mode Only**: Using Pass 1 fast scan only (fastest)")

            # Save settings button
            if st.button("💾 Save Settings", key="save_detailed_settings"):
                # Update config with session state values using __dict__ approach first
                try:
                    # Primary method: use __dict__ to bypass Pydantic validation
                    config.processing.__dict__["use_detailed_analysis"] = (
                        st.session_state.detailed_analysis_settings[
                            "use_detailed_analysis"
                        ]
                    )
                    config.processing.__dict__["detail_level"] = (
                        st.session_state.detailed_analysis_settings["detail_level"]
                    )
                    config.processing.__dict__["context_window_size"] = (
                        st.session_state.detailed_analysis_settings[
                            "context_window_size"
                        ]
                    )
                    config.processing.__dict__["analysis_methods"] = (
                        st.session_state.detailed_analysis_settings["analysis_methods"]
                    )

                    # Try direct assignment as backup (in case __dict__ approach fails)
                    try:
                        config.processing.use_detailed_analysis = (
                            st.session_state.detailed_analysis_settings[
                                "use_detailed_analysis"
                            ]
                        )
                        config.processing.detail_level = (
                            st.session_state.detailed_analysis_settings["detail_level"]
                        )
                        config.processing.context_window_size = (
                            st.session_state.detailed_analysis_settings[
                                "context_window_size"
                            ]
                        )
                        config.processing.analysis_methods = (
                            st.session_state.detailed_analysis_settings[
                                "analysis_methods"
                            ]
                        )
                    except (AttributeError, ValueError):
                        # If direct assignment fails, __dict__ approach already worked
                        pass

                except Exception as e:
                    st.error(f"Failed to save settings: {e}")
                    # Don't save if there were errors
                    st.stop()

                config.save_to_file()
                st.success("Settings saved!")

            # Output settings
            st.subheader("Output Settings")
            # Output Directory - More Intuitive Version
            st.markdown("**📁 Output Directory**")

            # Quick directory options
            import os

            current_dir = os.getcwd()
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")

            dir_options = {
                "📍 Current Directory": current_dir,
                "📁 Custom Path": "custom",
            }

            selected_option = st.selectbox(
                "Choose output location:",
                options=list(dir_options.keys()),
                index=0,
                help="Select where to save processed videos",
                key="output_dir_option",
            )

            if selected_option == "📁 Custom Path":
                output_dir = st.text_input(
                    "Custom path:",
                    value=(
                        st.session_state.output_dir
                        if st.session_state.output_dir
                        else ""
                    ),
                    placeholder=f"e.g., {current_dir}/processed_videos",
                    help="Enter full path to output directory",
                    key="output_dir_custom",
                )
                # Validate custom path
                if output_dir and not os.path.isabs(output_dir):
                    st.warning("⚠️ Please enter a full path (starting with / or C:)")
            else:
                output_dir = dir_options[selected_option]
                st.info(f"📍 Selected: {output_dir}")

            # Show current selection
            if output_dir:
                if os.path.exists(output_dir):
                    st.success(f"✅ Directory exists: {output_dir}")
                else:
                    st.info(f"📂 Directory will be created: {output_dir}")

            # Initialize output_dir to prevent NameError
            output_dir = st.session_state.get("output_dir", current_dir)

            # Auto-create directory if it doesn't exist
            if output_dir and not os.path.exists(output_dir):
                if st.button("📁 Create Directory", help="Create the output directory"):
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        st.success(f"✅ Created: {output_dir}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to create directory: {e}")

            # Store in session state for use across methods
            st.session_state.output_dir = output_dir

            # Duration settings
            st.subheader("Video Durations")
            self.generate_60s = st.checkbox("60 Second Video", value=True)
            self.generate_12min = st.checkbox("12 Minute Video", value=True)
            self.generate_1h = st.checkbox("1 Hour Video", value=True)
            self.custom_duration = st.number_input(
                "Custom Duration (seconds)", min_value=10, max_value=3600, value=120
            )

            # Save settings button
            if st.button("💾 Save Settings", key="save_general_settings"):
                config.save_to_file()
                st.success("Settings saved!")

            # Processing State Settings
            st.divider()
            st.subheader("💾 Processing State")
            st.markdown("**📁 Processing State Directory**")

            # Default processing state directory
            default_state_dir = os.path.join(os.getcwd(), "processing_states")
            if (
                not hasattr(config.processing, "processing_state_dir")
                or config.processing.processing_state_dir is None
            ):
                config.processing.__dict__["processing_state_dir"] = Path(
                    default_state_dir
                )

            state_dir_input = st.text_input(
                "Processing state directory:",
                value=str(config.processing.processing_state_dir),
                placeholder=default_state_dir,
                help="Directory to save processing states for resume functionality",
            )

            if st.button(
                "📁 Create State Directory",
                help="Create the processing state directory",
            ):
                try:
                    os.makedirs(state_dir_input, exist_ok=True)
                    config.processing.__dict__["processing_state_dir"] = Path(
                        state_dir_input
                    )
                    st.success(f"✅ Created: {state_dir_input}")
                except Exception as e:
                    st.error(f"❌ Failed to create directory: {e}")

            # Enable resume functionality
            config.processing.__dict__["enable_resume"] = st.checkbox(
                "Enable Resume Functionality",
                value=getattr(config.processing, "enable_resume", True),
                help="Save processing state to resume interrupted processing",
            )

    def _render_main_content(self):
        """Render main content area"""
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📹 Process Videos", "📊 Analytics", "🎵 Audio", "📤 Upload"]
        )

        with tab1:
            self._render_video_processing_tab()
        with tab2:
            self._render_analytics_tab()
        with tab3:
            self._render_audio_tab()
        with tab4:
            self._render_upload_tab()

    def _render_video_processing_tab(self):
        """Render video processing tab"""
        st.header("Video Processing")

        # File upload
        uploaded_files = st.file_uploader(
            "Choose video files",
            type=["mp4", "avi", "mkv", "mov", "wmv"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

            # Display uploaded files
            st.subheader("Uploaded Files")
            for file in uploaded_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📹 {file.name}")
                with col2:
                    st.write(f"{file.size / (1024*1024):.1f} MB")
                with col3:
                    if st.button("❌ Remove", key=f"remove_{file.name}"):
                        st.session_state.uploaded_files.remove(file)
                        st.rerun()

            # Processing options
            st.subheader("Processing Options")
            col1, col2 = st.columns(2)

            with col1:
                output_format = st.selectbox(
                    "Output Format", ["mp4", "avi", "mkv", "mov", "wmv"], index=0
                )

                # output_dir = st.text_input(
                #     "Output Directory",
                #     value=str(config.output_dir) if config.output_dir else "",
                # )

            with col2:
                add_watermark = st.checkbox("Add Watermark")
                watermark_text = st.text_input(
                    "Watermark Text",
                    value=config.upload.privacy_status if add_watermark else "",
                )

            # Capture settings BEFORE processing starts
            use_optimized = st.checkbox(
                "💾 Memory-Efficient Mode",
                value=True,
                help="Use less memory (recommended for large files)",
            )
            use_gpu = st.checkbox(
                "🚀 Use GPU Acceleration",
                value=True,
                help="Enable GPU processing for faster results",
            )

            # Start processing button - DISABLED during processing
            is_processing = (
                st.session_state.current_job is not None
                and st.session_state.current_job.get("status") == "running"
            )

            if st.button(
                "🚀 Start Processing" if not is_processing else "⏳ Processing...",
                type="primary",
                disabled=is_processing or len(st.session_state.uploaded_files) == 0,
            ):
                self._start_processing(
                    st.session_state.uploaded_files,
                    output_format,
                    st.session_state.output_dir,
                    add_watermark,
                    watermark_text,
                    use_optimized,  # Pass settings
                    use_gpu,
                )

        # Check for resumable processing
        if getattr(config.processing, "enable_resume", True):
            state_dir = getattr(config.processing, "processing_state_dir", None)
            if state_dir and Path(state_dir).exists():
                state_files = list(Path(state_dir).glob("processing_state_*.json"))
                if state_files:
                    st.subheader("🔄 Resume Processing")
                    st.info("Found interrupted processing sessions:")

                    for state_file in sorted(
                        state_files, key=lambda x: x.stat().st_mtime, reverse=True
                    ):
                        try:
                            with open(state_file, "r") as f:
                                state_data = json.load(f)

                            timestamp = datetime.fromisoformat(state_data["timestamp"])
                            st.write(
                                f"📁 Session from {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                            )

                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(
                                    f"Progress: {state_data['current_file_index'] + 1}/{len(st.session_state.uploaded_files) if st.session_state.uploaded_files else '?'} files"
                                )
                            with col2:
                                if st.button(
                                    "🔄 Resume", key=f"resume_{state_file.name}"
                                ):
                                    self._resume_processing(state_data)
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error loading state file: {e}")

        # Processing progress
        if st.session_state.current_job:
            self._render_processing_progress()

    def _start_processing(
        self,
        files,
        output_format,
        output_dir,
        add_watermark,
        watermark_text,
        use_optimized,
        use_gpu,
    ):
        """Start video processing job with proper callback handling"""

        # Initialize job with settings captured
        st.session_state.current_job = {
            "status": "running",
            "progress": 0.0,  # Changed from 0 to 0.0
            "current_file": "",
            "start_time": time.time(),
            "logs": [],  # For real-time logging
            "settings": {
                "use_optimized": use_optimized,
                "use_gpu": use_gpu,
                "output_format": output_format,
            },
        }

        # Force UI refresh to show updated button state
        # Add a placeholder that will trigger UI update
        if not hasattr(st.session_state, "button_state_placeholder"):
            st.session_state.button_state_placeholder = st.empty()
        st.session_state.button_state_placeholder.text("🔄 Processing started...")

        # Create progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        elapsed_text = st.empty()

        def progress_callback(progress, message, debug_info=None):
            """Enhanced progress callback with debug support"""
            try:
                # Validate and convert progress
                progress = self._validate_progress(progress)

                # Convert to 0.0-1.0 range for st.progress()
                normalized_progress = progress / 100.0

                # Update job progress
                st.session_state.current_job["progress"] = normalized_progress

            except Exception as e:
                # Fallback to 0.0 if conversion fails
                st.session_state.current_job["progress"] = 0.0
                logger.warning(f"Failed to convert progress value '{progress}': {e}")

            if isinstance(message, str):
                # Extract file info from message if present
                if "Processing:" in message:
                    current_file = message.split("Processing:")[1].split("(")[0].strip()
                    st.session_state.current_job["current_file"] = current_file

                # Add to logs
                st.session_state.current_job["logs"].append(f"{time.time()}: {message}")

            # Handle debug information
            if debug_info and isinstance(debug_info, dict):
                st.session_state.current_job.update(debug_info)

            # Update UI elements with error handling
            try:
                # Ensure progress is always a valid float between 0.0 and 1.0
                current_progress = st.session_state.current_job["progress"]
                if (
                    not isinstance(current_progress, float)
                    or current_progress < 0.0
                    or current_progress > 1.0
                ):
                    current_progress = 0.0

                progress_bar.progress(current_progress)
                status_text.text(message)

                # Show elapsed time during processing
                elapsed = time.time() - st.session_state.current_job["start_time"]
                elapsed_text.text(f"⏱️ Elapsed: {elapsed:.1f}s")

                # Force UI refresh by updating a placeholder
                # This ensures the debug information updates in real-time
                if hasattr(st.session_state, "debug_placeholder"):
                    st.session_state.debug_placeholder.empty()

                # Also refresh the button state placeholder
                if hasattr(st.session_state, "button_state_placeholder"):
                    st.session_state.button_state_placeholder.text(
                        f"⏳ Processing: {message}"
                    )

            except Exception as e:
                logger.warning(f"Failed to update progress UI: {e}")

        try:
            # Process videos with captured settings
            results = []
            total_files = len(files)

            for i, uploaded_file in enumerate(files):
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_path = f"temp_{uploaded_file.name}"

                progress_callback(
                    (i / total_files) * 100,
                    f"📹 Processing: {uploaded_file.name}",
                    {"current_file": uploaded_file.name},
                )

                # Use captured settings (no UI elements in processing loop)
                if use_optimized:
                    result = self.processor.process_video_streaming(
                        file_path,
                        output_format,
                        st.session_state.output_dir,
                        watermark_text,
                        progress_callback,
                        use_gpu=use_gpu,
                    )
                else:
                    result = self.processor.process_video(
                        file_path,
                        output_format,
                        st.session_state.output_dir,
                        watermark_text,
                        progress_callback,
                    )

                results.append(result)
                Path(file_path).unlink(missing_ok=True)

            # Save stats after completion
            self._save_stats()

            st.success("✅ Processing completed!")

        except Exception as e:
            st.session_state.current_job["status"] = "error"
            st.session_state.current_job["error"] = str(e)
            logger.error(f"Processing failed: {e}")

            # Save error stats
            self._save_stats()

    def _save_processing_state(self, job_id, current_file_index, processed_files):
        """Save current processing state for resume functionality"""
        if not getattr(config.processing, "enable_resume", True):
            return

        state_dir = getattr(config.processing, "processing_state_dir", None)
        if not state_dir:
            return

        state_file = Path(state_dir) / f"processing_state_{job_id}.json"

        state_data = {
            "job_id": job_id,
            "current_file_index": current_file_index,
            "processed_files": processed_files,
            "settings": st.session_state.current_job.get("settings", {}),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            Path(state_dir).mkdir(parents=True, exist_ok=True)
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)
            st.session_state.current_job["state_file"] = str(state_file)
        except Exception as e:
            logger.warning(f"Failed to save processing state: {e}")

    def _load_processing_state(self, job_id):
        """Load saved processing state"""
        state_dir = getattr(config.processing, "processing_state_dir", None)
        if not state_dir:
            return None

        state_file = Path(state_dir) / f"processing_state_{job_id}.json"

        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load processing state: {e}")

        return None

    def _resume_processing(self, state_data):
        """Resume processing from saved state"""
        st.info(
            f"🔄 Resuming from file {state_data['current_file_index'] + 1} of {len(st.session_state.uploaded_files)}"
        )

        # Restore settings
        st.session_state.current_job = {
            "status": "running",
            "progress": 0.0,
            "current_file": "",
            "start_time": time.time(),
            "logs": [],
            "settings": state_data["settings"],
            "resumed": True,
            "resume_from": state_data["current_file_index"],
        }

        # Continue from where we left off
        self._start_processing(
            st.session_state.uploaded_files[state_data["current_file_index"] :],
            state_data["settings"].get("output_format", "mp4"),
            st.session_state.output_dir,
            False,  # add_watermark - would need to save this
            "",  # watermark_text - would need to save this
            state_data["settings"].get("use_optimized", True),
            state_data["settings"].get("use_gpu", True),
        )

    def _render_processing_progress(self):
        """Enhanced processing progress with real-time updates"""
        job = st.session_state.current_job

        if job["status"] == "running":
            # Add a placeholder for forcing UI updates
            if not hasattr(st.session_state, "debug_placeholder"):
                st.session_state.debug_placeholder = st.empty()

            # Update the placeholder to force refresh
            st.session_state.debug_placeholder.text(f"Last update: {time.time()}")

            # Safely get and validate progress value
            progress_value = job.get("progress", 0.0)

            try:
                if isinstance(progress_value, str):
                    progress_value = 0.0
                elif not isinstance(progress_value, (int, float)):
                    progress_value = 0.0

                # Ensure progress is in valid range
                progress_value = max(0.0, min(1.0, float(progress_value)))

                st.progress(progress_value)
            except Exception as e:
                # Fallback progress bar
                st.progress(0.0)
                logger.warning(f"Failed to display progress: {e}")

            st.info(f"📹 Processing: {job.get('current_file', 'Unknown')}")

            # Show elapsed time during processing (not just after)
            try:
                elapsed = time.time() - job.get("start_time", time.time())
                st.text(f"⏱️ Elapsed: {elapsed:.1f}s")
            except Exception as e:
                st.text("⏱️ Elapsed: 0.0s")
                logger.warning(f"Failed to calculate elapsed time: {e}")

            # Debug Information - Always visible during processing
            st.subheader("🔍 Debug Information")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 System Resources**")

                if HAS_PSUTIL:
                    try:
                        memory = psutil.virtual_memory()
                        st.metric("RAM Usage", f"{memory.percent:.1f}%")
                        st.metric(
                            "Available RAM", f"{memory.available / (1024**3):.1f}GB"
                        )
                    except:
                        st.text("Memory monitoring unavailable")

                try:
                    import torch

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        gpu_used = torch.cuda.memory_allocated(0)
                        gpu_percent = (gpu_used / gpu_memory) * 100
                        st.metric("GPU Memory", f"{gpu_percent:.1f}%")
                    else:
                        st.text("GPU not available")
                except:
                    st.text("GPU monitoring unavailable")

            with col2:
                st.markdown("**⚙️ Processing Details**")

                st.text(f"Stage: {job.get('stage', 'Processing')}")

                current_frame = job.get("current_frame", 0)
                total_frames = job.get("total_frames", 0)
                if total_frames > 0:
                    st.metric("Frame Progress", f"{current_frame}/{total_frames}")

                settings = job.get("settings", {})
                st.text(f"Memory-Efficient: {settings.get('use_optimized', True)}")
                st.text(f"GPU Acceleration: {settings.get('use_gpu', True)}")
                st.text(f"Output Format: {settings.get('output_format', 'mp4')}")

            # Real-time logs
            st.markdown("**📝 Real-time Logs**")
            logs = job.get("logs", [])
            if logs:
                # Show last 10 log entries
                try:
                    log_text = "\n".join([log.split(": ", 1)[1] for log in logs[-10:]])
                    st.text_area(
                        "Recent Logs",
                        log_text,
                        height=150,
                        disabled=True,
                        key="debug_logs",
                    )
                except:
                    st.text_area(
                        "Recent Logs",
                        "Log parsing error",
                        height=150,
                        disabled=True,
                        key="debug_logs_error",
                    )
            else:
                st.text("No logs available yet")

        elif job["status"] == "completed":
            st.success("✅ Processing completed!")

            # Debug information in expandable menu after completion
            with st.expander("🔍 Processing Details & Logs", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📊 Final System Resources**")

                    if HAS_PSUTIL:
                        try:
                            memory = psutil.virtual_memory()
                            st.metric("Final RAM Usage", f"{memory.percent:.1f}%")
                            st.metric(
                                "Available RAM", f"{memory.available / (1024**3):.1f}GB"
                            )
                        except:
                            st.text("Memory monitoring unavailable")

                    try:
                        import torch

                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.get_device_properties(
                                0
                            ).total_memory
                            gpu_used = torch.cuda.memory_allocated(0)
                            gpu_percent = (gpu_used / gpu_memory) * 100
                            st.metric("Final GPU Memory", f"{gpu_percent:.1f}%")
                        else:
                            st.text("GPU not available")
                    except:
                        st.text("GPU monitoring unavailable")

                with col2:
                    st.markdown("**⚙️ Processing Summary**")

                    total_frames = job.get("total_frames", 0)
                    if total_frames > 0:
                        st.metric("Total Frames", total_frames)

                    settings = job.get("settings", {})
                    st.text(f"Memory-Efficient: {settings.get('use_optimized', True)}")
                    st.text(f"GPU Acceleration: {settings.get('use_gpu', True)}")
                    st.text(f"Output Format: {settings.get('output_format', 'mp4')}")

                # Processing logs
                st.markdown("**📝 Processing Logs**")
                logs = job.get("logs", [])
                if logs:
                    try:
                        log_text = "\n".join(
                            [log.split(": ", 1)[1] for log in logs[-20:]]
                        )
                        st.text_area(
                            "Complete Logs", log_text, height=200, disabled=True
                        )
                    except:
                        st.text_area(
                            "Complete Logs",
                            "Log parsing error",
                            height=200,
                            disabled=True,
                        )
                else:
                    st.text("No logs available")

            if "results" in job:
                self._display_results(job["results"])

        elif job["status"] == "error":
            st.error(f"❌ Error: {job.get('error', 'Unknown error')}")

            # Debug information in expandable menu for error cases
            with st.expander("🔍 Error Details & Logs", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📊 System State at Error**")

                    if HAS_PSUTIL:
                        try:
                            memory = psutil.virtual_memory()
                            st.metric("RAM Usage at Error", f"{memory.percent:.1f}%")
                        except:
                            st.text("Memory monitoring unavailable")

                with col2:
                    st.markdown("**⚙️ Error Context**")

                    st.text(f"Stage: {job.get('stage', 'Unknown')}")
                    settings = job.get("settings", {})
                    st.text(f"Memory-Efficient: {settings.get('use_optimized', True)}")
                    st.text(f"GPU Acceleration: {settings.get('use_gpu', True)}")

                # Error logs
                st.markdown("**📝 Error Logs**")
                logs = job.get("logs", [])
                if logs:
                    try:
                        log_text = "\n".join(
                            [log.split(": ", 1)[1] for log in logs[-20:]]
                        )
                        st.text_area("Error Logs", log_text, height=200, disabled=True)
                    except:
                        st.text_area(
                            "Error Logs",
                            "Log parsing error",
                            height=200,
                            disabled=True,
                        )
                else:
                    st.text("No logs available")

    def _display_results(self, results):
        """Display processing results"""
        st.subheader("Processing Results")

        for result in results:
            with st.expander(f"📹 {result.filename}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Frames Processed", result.frames_processed)
                with col2:
                    st.metric("Motion Events", result.motion_events)
                with col3:
                    st.metric("Processing Time", f"{result.processing_time:.1f}s")

                # Display output files
                if result.output_files:
                    st.write("**Output Files:**")
                    for output_file in result.output_files:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"📄 {Path(output_file).name}")
                        with col2:
                            if st.button("📤 Upload", key=f"upload_{output_file}"):
                                self._upload_to_youtube(output_file)

    def _render_analytics_tab(self):
        """Render analytics tab"""
        st.header("Analytics Dashboard")

        # Get analytics data
        analytics_data = self.analytics_service.get_analytics()

        if not analytics_data:
            st.info("No analytics data available yet.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_videos = len(analytics_data.get("videos_processed", []))
            st.metric("Total Videos", total_videos)

        with col2:
            total_frames = sum(
                v.get("frames_processed", 0)
                for v in analytics_data.get("videos_processed", [])
            )
            st.metric("Total Frames", total_frames)

        with col3:
            avg_processing_time = analytics_data.get("avg_processing_time", 0)
            st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")

        with col4:
            success_rate = analytics_data.get("success_rate", 0) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Processing time chart
            if analytics_data.get("videos_processed"):
                processing_times = [
                    v.get("processing_time", 0)
                    for v in analytics_data["videos_processed"]
                ]
                fig = px.histogram(
                    x=processing_times,
                    title="Processing Time Distribution",
                    labels={"x": "Processing Time (s)", "y": "Frequency"},
                )
                st.plotly_chart(fig)

        with col2:
            # Motion events chart
            if analytics_data.get("videos_processed"):
                motion_events = [
                    v.get("motion_events", 0)
                    for v in analytics_data["videos_processed"]
                ]
                fig = px.scatter(
                    x=processing_times,
                    y=motion_events,
                    title="Processing Time vs Motion Events",
                    labels={"x": "Processing Time (s)", "y": "Motion Events"},
                )
                st.plotly_chart(fig)

        # Recent processing history
        st.subheader("Recent Processing History")
        # Convert ProcessingResult objects to dictionaries
        history_data = []
        for item in st.session_state.processing_history:
            if hasattr(item, "results") and item.results:
                for result in item.results:
                    history_data.append(
                        {
                            "timestamp": item.get("timestamp", ""),
                            "filename": result.filename,
                            "frames_processed": result.frames_processed,
                            "motion_events": result.motion_events,
                            "processing_time": result.processing_time,
                            "output_files": len(result.output_files),
                            "error": result.error,
                        }
                    )

        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df)

    def _render_audio_tab(self):
        """Render audio settings tab with toggle and save functionality"""
        st.header("Audio Settings")

        # Audio toggle
        st.subheader("🎵 Audio Processing")
        enable_audio = st.checkbox(
            "Enable Audio Processing",
            value=getattr(config.audio, "enable_audio", True),
            help="Enable or disable audio processing (faster without audio)",
        )
        config.audio.__dict__["enable_audio"] = enable_audio

        if config.audio.__dict__.get("enable_audio", True):
            st.subheader("Background Music")

            # Default music
            st.write("**Default Music:**")
            default_music = st.file_uploader(
                "Upload default background music",
                type=["mp3", "wav", "ogg"],
                key="default_music",
            )
            if default_music:
                config.audio.music_paths["default"] = self.file_service.save_music_file(
                    default_music
                )
                config.audio.__dict__["selected_music"] = getattr(
                    config.audio, "selected_music", {}
                )
                config.audio.__dict__["selected_music"]["default"] = (
                    config.audio.music_paths["default"]
                )

            # Duration-specific music
            durations = {"60s": 60, "12min": 720, "1h": 3600}

            for label, seconds in durations.items():
                st.write(f"**Music for {label} videos:**")
                music_file = st.file_uploader(
                    f"Upload music for {label} videos",
                    type=["mp3", "wav", "ogg"],
                    key=f"music_{seconds}",
                )
                if music_file:
                    saved_path = self.file_service.save_music_file(music_file)
                    config.audio.music_paths[str(seconds)] = saved_path
                    config.audio.__dict__["selected_music"][str(seconds)] = (
                        config.audio.music_paths[selected]
                    )

            # Volume control
            st.subheader("Volume Control")
            config.audio.volume = st.slider(
                "Music Volume",
                min_value=0.0,
                max_value=2.0,
                value=config.audio.volume,
                step=0.1,
            )

            # Music selection for each duration
            st.subheader("🎵 Music Selection")
            for label, seconds in durations.items():
                available_music = list(config.audio.music_paths.keys())
                if available_music:
                    selected = st.selectbox(
                        f"Music for {label} videos",
                        options=available_music,
                        index=(
                            available_music.index(
                                config.audio.selected_music.get(str(seconds), "default")
                            )
                            if config.audio.selected_music.get(str(seconds))
                            in available_music
                            else 0
                        ),
                        key=f"select_music_{seconds}",
                    )
                    config.audio.selected_music[str(seconds)] = (
                        config.audio.music_paths[selected]
                    )
        else:
            st.info(
                "⚡ **Audio processing disabled** - Videos will be processed without background music"
            )

        # Save settings button
        if st.button("💾 Save Audio Settings", key="save_audio_settings"):
            config.save_to_file()
            st.success("Audio settings saved!")

    def _render_upload_tab(self):
        """Render YouTube upload tab"""
        st.header("YouTube Upload")

        # YouTube authentication
        st.subheader("YouTube Authentication")

        if not self.youtube_service.is_authenticated():
            st.warning("You need to authenticate with YouTube first.")
            if st.button("🔐 Authenticate with YouTube"):
                self.youtube_service.authenticate()
                st.rerun()
        else:
            st.success("✅ YouTube authenticated!")

            # Upload settings
            st.subheader("Upload Settings")

            privacy_options = ["public", "private", "unlisted"]
            config.upload.privacy_status = st.selectbox(
                "Privacy Status",
                privacy_options,
                index=privacy_options.index(config.upload.privacy_status),
            )

            config.upload.max_retries = st.slider(
                "Max Retries",
                min_value=1,
                max_value=20,
                value=config.upload.max_retries,
            )

            if st.button("💾 Save Upload Settings"):
                config.save_to_file()
                st.success("Upload settings saved!")

        # Manual upload section
        st.subheader("Manual Upload")

        upload_file = st.text_input("Enter path to video file for upload:")
        title = st.text_input("Video Title:")
        description = st.text_area("Video Description:")

        if st.button("📤 Upload to YouTube") and upload_file and title:
            if Path(upload_file).exists():
                self._upload_to_youtube(upload_file, title, description)
            else:
                st.error("File not found!")

    def _upload_to_youtube(self, file_path, title=None, description=None):
        """Upload video to YouTube"""
        if not title:
            title = Path(file_path).stem

        if not description:
            description = "Uploaded via NestCam Processor v2.0"

        try:
            with st.spinner("Uploading to YouTube..."):
                upload_progress = st.progress(0)

                def progress_callback(progress):
                    try:
                        # Validate progress using the same logic
                        if isinstance(progress, str):
                            numeric_match = re.search(r"(\d+\.?\d*)", progress)
                            if numeric_match:
                                progress = float(numeric_match.group(1))
                            else:
                                progress = 0.0
                        elif not isinstance(progress, (int, float)):
                            progress = 0.0

                        progress = max(0.0, min(100.0, float(progress)))
                        upload_progress.progress(progress / 100)
                    except Exception as e:
                        upload_progress.progress(0.0)
                        logger.warning(f"YouTube upload progress error: {e}")

                video_url = self.youtube_service.upload_video(
                    file_path, title, description, progress_callback=progress_callback
                )

                st.success(f"✅ Video uploaded successfully!")
                st.write(f"📺 Watch here: {video_url}")

        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            logger.error(f"YouTube upload failed: {e}")


def main():
    """Main application entry point"""
    app = NestCamApp()
    app.run()


if __name__ == "__main__":
    main()
