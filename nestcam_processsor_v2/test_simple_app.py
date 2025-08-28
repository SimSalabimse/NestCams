#!/usr/bin/env python3
"""
Simple test app to verify Streamlit works
"""

import streamlit as st


def main():
    st.set_page_config(page_title="🐦 NestCam Test", page_icon="🐦", layout="wide")

    st.title("🐦 NestCam Processor - Test Version")
    st.markdown("*Testing if Streamlit works*")

    st.success("✅ Streamlit is working!")

    # Basic system info
    import platform

    st.info(f"Platform: {platform.system()}")

    # Test GPU detection
    try:
        import torch

        if torch.cuda.is_available():
            st.info("🎯 CUDA GPU detected")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            st.info("🍎 Apple Silicon GPU detected")
        else:
            st.info("💻 CPU only")
    except ImportError:
        st.warning("⚠️ PyTorch not available")

    # Simple file uploader test
    st.subheader("📁 File Upload Test")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
