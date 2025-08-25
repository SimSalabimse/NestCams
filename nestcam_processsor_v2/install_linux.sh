#!/bin/bash
# NestCam Processor v2.0 - Linux Installation Script with Proper Venv Handling

set -e  # Exit on any error

echo "🐦 NestCam Processor v2.0 - Linux Installation"
echo "==============================================="

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/nestcam_env"

echo "📂 Project directory: $PROJECT_DIR"
echo "🌐 Virtual environment: $VENV_DIR"

# Update package list
echo "🔄 Updating package list..."
sudo apt update

# Install Python 3.11 and pip
echo "🐍 Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-pip python3.11-venv

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0

# Clean up existing venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo "🧹 Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
echo "🌐 Creating Python virtual environment..."
python3.11 -m venv "$VENV_DIR"

# Verify venv creation
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔗 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify we're in venv
if [[ "$VIRTUAL_ENV" != "$VENV_DIR" ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip within venv
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for Linux)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
echo "📚 Installing project dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Test installation
echo "🧪 Testing installation..."
python -c "
import sys
print(f'Python version: {sys.version}')
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    import cv2
    print('OpenCV available')
    import streamlit
    print('Streamlit available')
    print('✅ All dependencies installed successfully')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
    sys.exit(1)
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 To run the application:"
echo "   cd \"$PROJECT_DIR\""
echo "   source \"$VENV_DIR/bin/activate\""
echo "   python -m src.main --web"
echo ""
echo "📝 To deactivate the virtual environment later:"
echo "   deactivate"
