#!/bin/bash
# NestCam Processor v2.0 - Mac Installation Script with Proper Venv Handling

set -e  # Exit on any error

echo "🐦 NestCam Processor v2.0 - Mac Installation"
echo "=============================================="

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/nestcam_env"

echo "📂 Project directory: $PROJECT_DIR"
echo "🌐 Virtual environment: $VENV_DIR"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
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

# Install PyTorch with Metal support
echo "🔥 Installing PyTorch with Metal support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install project dependencies
echo "📚 Installing project dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Install additional GPU packages for Mac
echo "🍎 Installing Mac-specific GPU packages..."
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Test GPU detection
echo "🧪 Testing GPU detection..."
python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('✅ Metal GPU acceleration available')
        device = torch.device('mps')
        test_tensor = torch.randn(100, 100, device=device)
        print('✅ Metal device test passed')
    else:
        print('⚠️ Metal not available, using CPU')
except Exception as e:
    print(f'❌ GPU test failed: {e}')
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
echo ""
echo "🔧 If GPU acceleration doesn't work, try:"
echo "   source \"$VENV_DIR/bin/activate\""
echo "   pip install --upgrade torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cpu"

# Create helper activation script
echo "📝 Creating activation helper script..."
cat > "$PROJECT_DIR/activate.sh" << 'EOF'
#!/bin/bash
# NestCam Processor v2.0 - Quick Activation Script

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/nestcam_env"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the installation script first:"
    echo "   ./install_mac.sh"
    exit 1
fi

echo "🔗 Activating NestCam virtual environment..."
source "$VENV_DIR/bin/activate"

if [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    echo ""
    echo "🚀 To run the application:"
    echo "   python -m src.main --web"
    echo ""
    echo "📝 To deactivate later: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
EOF

chmod +x "$PROJECT_DIR/activate.sh"
