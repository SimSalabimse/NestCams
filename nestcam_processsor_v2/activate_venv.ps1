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
```

---

## 2. Linux Install Script

**Replace the entire file with this improved version:**

```
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
```

---

## 3. Windows Install Script

**Replace the entire file with this improved version:**

```powershell
# NestCam Processor v2.0 - Windows Installation Script with Proper Venv Handling

param(
    [switch]$CleanInstall = $false
)

Write-Host "🐦 NestCam Processor v2.0 - Windows Installation" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Get the project directory
$PROJECT_DIR = Split-Path -Parent $PSCommandPath
$VENV_DIR = Join-Path $PROJECT_DIR "nestcam_env"

Write-Host "📂 Project directory: $PROJECT_DIR" -ForegroundColor Cyan
Write-Host "🌐 Virtual environment: $VENV_DIR" -ForegroundColor Cyan

# Function to check if command exists
function Test-Command {
    param($Command)
    return Get-Command $Command -ErrorAction SilentlyContinue
}

# Check if Python 3.11 is installed
$python311 = Test-Command python3.11
if (-not $python311) {
    Write-Host "🐍 Installing Python 3.11..." -ForegroundColor Yellow

    # Download and install Python 3.11
    $pythonUrl = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
    $installerPath = "$env:TEMP\python311.exe"

    try {
        Write-Host "Downloading Python 3.11..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing

        Write-Host "Installing Python 3.11..." -ForegroundColor Yellow
        Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_venv=1" -Wait -NoNewWindow

        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

        Remove-Item $installerPath -ErrorAction SilentlyContinue
    } catch {
        Write-Host "❌ Failed to install Python 3.11: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Python 3.11 already installed" -ForegroundColor Green
}

# Install ffmpeg
Write-Host "🎬 Installing ffmpeg..." -ForegroundColor Yellow
try {
    # Download ffmpeg
    $ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $ffmpegZip = "$env:TEMP\ffmpeg.zip"
    $ffmpegDir = "$env:TEMP\ffmpeg"

    Write-Host "Downloading ffmpeg..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip -UseBasicParsing

    Write-Host "Extracting ffmpeg..." -ForegroundColor Yellow
    if (Test-Path $ffmpegDir) {
        Remove-Item $ffmpegDir -Recurse -Force
    }
    Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir

    # Find ffmpeg executable
    $ffmpegPath = Get-ChildItem -Path $ffmpegDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    if ($ffmpegPath) {
        $ffmpegBinDir = Split-Path $ffmpegPath.FullName -Parent
        Write-Host "Adding ffmpeg to PATH: $ffmpegBinDir" -ForegroundColor Yellow
        $env:PATH += ";$ffmpegBinDir"
    }

    Remove-Item $ffmpegZip -ErrorAction SilentlyContinue
} catch {
    Write-Host "⚠️ Failed to install ffmpeg automatically. Please install it manually." -ForegroundColor Yellow
    Write-Host "Visit: https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

# Clean up existing venv if requested or if it exists
if ($CleanInstall -or (Test-Path $VENV_DIR)) {
    if (Test-Path $VENV_DIR) {
        Write-Host "🧹 Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item $VENV_DIR -Recurse -Force
    }
}

# Create virtual environment
Write-Host "🌐 Creating Python virtual environment..." -ForegroundColor Yellow
try {
    & python3.11 -m venv $VENV_DIR
} catch {
    Write-Host "❌ Failed to create virtual environment: $_" -ForegroundColor Red
    exit 1
}

# Verify venv creation
$activateScript = Join-Path $VENV_DIR "Scripts\activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "❌ Virtual environment creation failed - activate script not found" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "🔗 Activating virtual environment..." -ForegroundColor Yellow
try {
    & $activateScript
} catch {
    Write-Host "❌ Failed to activate virtual environment: $_" -ForegroundColor Red
    exit 1
}

# Verify we're in venv
if (-not $env:VIRTUAL_ENV) {
    Write-Host "❌ Virtual environment activation failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
try {
    & python -m pip install --upgrade pip
} catch {
    Write-Host "❌ Failed to upgrade pip: $_" -ForegroundColor Red
}

# Install PyTorch with CUDA support (if NVIDIA GPU is available)
Write-Host "🔥 Installing PyTorch..." -ForegroundColor Yellow
$nvidia = Get-WmiObject -Query "SELECT * FROM Win32_VideoController WHERE Name LIKE '%NVIDIA%'" -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "🎯 NVIDIA GPU detected, installing CUDA version..." -ForegroundColor Green
    & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "📺 No NVIDIA GPU detected, installing CPU version..." -ForegroundColor Yellow
    & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Install project dependencies
Write-Host "📚 Installing project dependencies..." -ForegroundColor Yellow
try {
    & pip install -r "$PROJECT_DIR\requirements.txt"
} catch {
    Write-Host "❌ Failed to install dependencies: $_" -ForegroundColor Red
    exit 1
}

# Test installation
Write-Host "🧪 Testing installation..." -ForegroundColor Yellow
try {
    & python -c "
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
} catch {
    Write-Host "❌ Installation test failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 To run the application:" -ForegroundColor Cyan
Write-Host "   cd `"$PROJECT_DIR`"" -ForegroundColor White
Write-Host "   & `"$VENV_DIR\Scripts\activate.ps1`"" -ForegroundColor White
Write-Host "   python -m src.main --web" -ForegroundColor White
Write-Host ""
Write-Host "📝 To deactivate the virtual environment later:" -ForegroundColor Cyan
Write-Host "   deactivate" -ForegroundColor White
```

---

## 4. Create a Helper Script for Easy Activation

**Create this new file: `activate_venv.sh`** (for Mac/Linux):
```bash
#!/bin/bash
# Helper script to activate the NestCam virtual environment

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/nestcam_env"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the installation script first:"
    echo "   ./install_mac.sh  (on Mac)"
    echo "   ./install_linux.sh  (on Linux)"
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
```

**Create this new file: `activate_venv.ps1`** (for Windows):
```powershell
<code_block_to_apply_changes_from>
```

---

## How to Use the Improved Install Scripts:

### Mac:
```bash
# Make scripts executable
chmod +x install_mac.sh
chmod +x activate_venv.sh

# Run installation
./install_mac.sh

# Later, to activate venv and run
./activate_venv.sh
python -m src.main --web
```

### Linux:
```bash
# Make scripts executable
chmod +x install_linux.sh
chmod +x activate_venv.sh

# Run installation
./install_linux.sh

# Later, to activate venv and run
./activate_venv.sh
python -m src.main --web
```

### Windows:
```powershell
# Run installation (with clean install if needed)
.\install_windows.ps1 -CleanInstall

# Later, to activate venv and run
.\activate_venv.ps1
python -m src.main --web
```

## Key Improvements Made:

1. **Proper Venv Handling**: All scripts now properly create, verify, and activate virtual environments
2. **Error Checking**: Added comprehensive error checking and validation
3. **Path Resolution**: Scripts automatically detect the correct project directory
4. **Clean Installation**: Option to clean existing venv before installation
5. **Installation Testing**: Scripts test that all dependencies are properly installed
6. **Helper Scripts**: Easy-to-use activation scripts for daily use
7. **Cross-Platform**: Consistent behavior across Mac, Linux, and Windows
8. **Better Instructions**: Clear instructions for activation and running the application

The scripts now ensure that all installations happen within the virtual environment, preventing system-wide package conflicts and ensuring consistent behavior across different platforms.
