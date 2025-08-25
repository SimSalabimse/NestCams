#!/bin/bash
# NestCam Processor v2.0 - Linux Installation Script

echo "🐦 NestCam Processor v2.0 - Linux Installation"
echo "==============================================="

# Update package list
echo "🔄 Updating package list..."
sudo apt update

# Install Python 3.11 and pip
echo "🐍 Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-pip python3.11-venv

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0

# Create virtual environment
echo "🌐 Creating Python virtual environment..."
python3.11 -m venv nestcam_env

# Activate virtual environment
echo "🔗 Activating virtual environment..."
source nestcam_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for Linux)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
echo "📚 Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 To run the application:"
echo "   source nestcam_env/bin/activate"
echo "   cd /Users/simsalabim/Documents/GitHub/NestCams/nestcam_processsor_v2"
echo "   python -m src.main --web"
