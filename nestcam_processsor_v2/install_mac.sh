#!/bin/bash
# NestCam Processor v2.0 - Mac Installation Script

echo "🐦 NestCam Processor v2.0 - Mac Installation"
echo "=============================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

# Update Homebrew
echo "🔄 Updating Homebrew..."
brew update

# Install Python 3.11 if not present
if ! command -v python3.11 &> /dev/null; then
    echo "🐍 Installing Python 3.11..."
    brew install python@3.11
else
    echo "✅ Python 3.11 already installed"
fi

# Install ffmpeg
echo "🎬 Installing ffmpeg..."
brew install ffmpeg

# Create virtual environment
echo "🌐 Creating Python virtual environment..."
python3.11 -m venv nestcam_env

# Activate virtual environment
echo "🔗 Activating virtual environment..."
source nestcam_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with Metal support
echo "🔥 Installing PyTorch with Metal support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install project dependencies
echo "📚 Installing project dependencies..."
pip install -r requirements.txt

# Install additional GPU packages for Mac
echo "🍎 Installing Mac-specific GPU packages..."
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 To run the application:"
echo "   source nestcam_env/bin/activate"
echo "   cd /Users/simsalabim/Documents/GitHub/NestCams/nestcam_processsor_v2"
echo "   python -m src.main --web"
echo ""
echo "🔧 If GPU acceleration doesn't work, try:"
echo "   pip install --upgrade torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cpu"
