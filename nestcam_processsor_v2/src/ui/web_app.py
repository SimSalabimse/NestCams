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
            if st.button("💾 Save Settings"):
                config.save_to_file()
                st.success("Settings saved!")

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

            # Start processing button
            if st.button("🚀 Start Processing", type="primary"):
                self._start_processing(
                    st.session_state.uploaded_files,
                    output_format,
                    st.session_state.output_dir,
                    add_watermark,
                    watermark_text,
                )

        # Processing progress
        if st.session_state.current_job:
            self._render_processing_progress()

    def _start_processing(
        self, files, output_format, output_dir, add_watermark, watermark_text
    ):
        """Start video processing job"""
        st.session_state.current_job = {
            "status": "running",
            "progress": 0,
            "current_file": "",
            "start_time": time.time(),
        }

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(progress, current_file, total_files):
            st.session_state.current_job["progress"] = progress / 100
            st.session_state.current_job["current_file"] = current_file
            progress_bar.progress(progress / 100)
            status_text.text(f"Processing: {current_file} ({int(progress)}%)")

        try:
            # Process videos
            results = []
            total_files = len(files)

            for i, uploaded_file in enumerate(files):
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_path = f"temp_{uploaded_file.name}"
                progress_callback(
                    (i / total_files) * 100, uploaded_file.name, total_files
                )

                # Use memory-efficient streaming processing
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

                # Choose processing method
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

                # Clean up temp file
                Path(file_path).unlink(missing_ok=True)

            st.session_state.current_job["status"] = "completed"
            st.session_state.current_job["results"] = results

            # Add to processing history
            st.session_state.processing_history.append(
                {
                    "timestamp": datetime.now(),
                    "files_processed": len(files),
                    "results": results,
                }
            )

            st.success("✅ Processing completed!")

        except Exception as e:
            st.session_state.current_job["status"] = "error"
            st.session_state.current_job["error"] = str(e)
            st.error(f"❌ Processing failed: {str(e)}")
            logger.error(f"Processing failed: {e}")

    def _render_processing_progress(self):
        """Render processing progress"""
        job = st.session_state.current_job

        st.subheader("Processing Progress")

        if job["status"] == "running":
            st.progress(job["progress"])
            st.info(f"📹 Processing: {job['current_file']}")

            elapsed = time.time() - job["start_time"]
            st.text(f"⏱️ Elapsed: {elapsed:.1f}s")

        elif job["status"] == "completed":
            st.success("✅ Processing completed!")
            if "results" in job:
                self._display_results(job["results"])

        elif job["status"] == "error":
            st.error(f"❌ Error: {job['error']}")

    def _display_results(self, results):
        """Display processing results"""
        st.subheader("Processing Results")

        for result in results:
            with st.expander(f"📹 {result['filename']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Frames Processed", result.get("frames_processed", 0))
                with col2:
                    st.metric("Motion Events", result.get("motion_events", 0))
                with col3:
                    st.metric(
                        "Processing Time", f"{result.get('processing_time', 0):.1f}s"
                    )

                # Display output files
                if "output_files" in result:
                    st.write("**Output Files:**")
                    for output_file in result["output_files"]:
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
        history_df = pd.DataFrame(st.session_state.processing_history)
        if not history_df.empty:
            st.dataframe(history_df)

    def _render_audio_tab(self):
        """Render audio settings tab"""
        st.header("Audio Settings")

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
            st.success("Default music uploaded!")

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
                config.audio.music_paths[str(seconds)] = (
                    self.file_service.save_music_file(music_file)
                )
                st.success(f"Music for {label} videos uploaded!")

        # Volume control
        st.subheader("Volume Control")
        config.audio.volume = st.slider(
            "Music Volume",
            min_value=0.0,
            max_value=2.0,
            value=config.audio.volume,
            step=0.1,
        )

        if st.button("💾 Save Audio Settings"):
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
                    upload_progress.progress(progress / 100)

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
