#!/bin/bash
set -e
echo "Bird Box Video Processor v12.0 — Setup"

command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
command -v ffmpeg  >/dev/null 2>&1 || {
    echo "FFmpeg not found."
    echo "  macOS:  brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    exit 1
}

[ -d venv ] || python3 -m venv venv
source venv/bin/activate

echo "Removing conflicting OpenCV builds..."
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
python3 -c "import cv2; cv2.ocl.setUseOpenCL(True); ok=cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL(); print('  OpenCL:', ok)"
echo ""
echo "Setup complete. Run: bash Start.sh"
