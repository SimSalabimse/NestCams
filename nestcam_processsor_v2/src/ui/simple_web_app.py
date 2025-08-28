#!/usr/bin/env python3
"""
Simplified NestCam web app for testing
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    st.set_page_config(
        page_title="🐦 NestCam Processor v3.0", page_icon="🐦", layout="wide"
    )

    st.title("🐦 NestCam Processor v3.0")
    st.markdown("*Simplified version for testing*")

    # Test basic imports
    st.subheader("🔧 System Status")

    try:
        from config import GPU_BACKEND, HAS_GPU

        st.success(f"✅ Config loaded - GPU: {GPU_BACKEND.upper()}")
    except Exception as e:
        st.error(f"❌ Config import failed: {e}")

    try:
        import torch

        st.success("✅ PyTorch available")
    except ImportError:
        st.warning("⚠️ PyTorch not available")

    # Simple file uploader
    st.subheader("📁 File Upload")
    uploaded_files = st.file_uploader(
        "Choose video files",
        type=["mp4", "avi", "mkv", "mov"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / (1024*1024):.1f} MB)")

    # Simple processing simulation
    if uploaded_files and st.button("🚀 Start Processing", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing: {file.name}")
            progress_bar.progress((i + 1) / len(uploaded_files))
            import time

            time.sleep(1)  # Simulate processing

        st.success("✅ Processing completed!")
        status_text.empty()


if __name__ == "__main__":
    main()
