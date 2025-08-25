# 🐦 NestCam Processor v2.0

Advanced Bird Nest Video Processing with GPU acceleration and AI-powered motion detection.

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Mac Installation](#mac-installation)
  - [Linux Installation](#linux-installation)
  - [Windows Installation](#windows-installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Development](#development)
- [License](#license)

## ✨ Features

- 🎯 **Advanced Motion Detection**: Two-pass analysis with configurable detail levels
- 🚀 **GPU Acceleration**: Metal (Mac), CUDA (NVIDIA), and OpenCV CUDA support
- 🎬 **Video Processing**: Multiple output formats and durations
- 🎵 **Audio Integration**: Background music support with volume control
- 📊 **Analytics Dashboard**: Processing statistics and performance metrics
- 💾 **Resume Functionality**: Save and resume interrupted processing
- 🌐 **Web Interface**: Streamlit-based GUI for easy operation
- 🔧 **Cross-Platform**: Mac, Linux, and Windows support

## 🔧 System Requirements

### Minimum Requirements

- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **Python**: 3.11+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Network**: Internet connection for installation

### Recommended Requirements

- **OS**: macOS 12.0+, Ubuntu 20.04+, Windows 11
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU (for CUDA) or Apple Silicon (for Metal)
- **Storage**: SSD with 50GB+ free space

## 🚀 Quick Start

### Option 1: Automated Installation

```bash
# Mac
./install_mac.sh

# Linux
./install_linux.sh

# Windows
.\install_windows.ps1
```

### Option 2: Manual Installation

```bash
# 1. Create virtual environment
python -m venv nestcam_env

# 2. Activate environment
source nestcam_env/bin/activate  # Mac/Linux
# OR
.\nestcam_env\Scripts\activate.ps1  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
python -m src.main --web
```

## 📦 Installation

### Mac Installation

#### Prerequisites

- [Homebrew](https://brew.sh/) (will be installed if missing)
- Command Line Tools for Xcode

#### Automated Installation

```bash
# Make script executable
chmod +x install_mac.sh

# Run installation
./install_mac.sh
```

#### Manual Installation

```bash
# Install system dependencies
brew install python@3.11 ffmpeg

# Create virtual environment
python3.11 -m venv nestcam_env

# Activate environment
source nestcam_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux Installation

#### Prerequisites

- sudo access
- apt package manager

#### Automated Installation

```bash
# Make script executable
chmod +x install_linux.sh

# Run installation
./install_linux.sh
```

#### Manual Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-pip python3.11-venv ffmpeg

# Create virtual environment
python3.11 -m venv nestcam_env

# Activate environment
source nestcam_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windows Installation

#### Prerequisites

- Windows 10 or later
- Administrator rights for installation
- PowerShell 5.1 or later

#### Automated Installation

```powershell
# Run installation (with verbose output)
.\install_windows.ps1 -Verbose

# For clean installation (removes existing venv)
.\install_windows.ps1 -CleanInstall -Verbose

# Skip tests during installation
.\install_windows.ps1 -SkipTests
```

#### Manual Installation

```powershell
# Install Python 3.11 from python.org
# Install ffmpeg from https://ffmpeg.org/download.html

# Create virtual environment
python3.11 -m venv nestcam_env

# Activate environment
.\nestcam_env\Scripts\activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Starting the Application

#### Using Helper Scripts

```bash
# Mac/Linux
./activate_venv.sh

# Windows
.\activate_venv.ps1
```

#### Manual Activation

```bash
# Mac/Linux
source nestcam_env/bin/activate
python -m src.main --web

# Windows
.\nestcam_env\Scripts\activate.ps1
python -m src.main --web
```

### Command Line Options

```bash
python -m src.main [OPTIONS]

Options:
  --web          Run web interface (default)
  --cli          Run command line interface
  --config PATH  Path to custom config file
  --debug        Enable debug mode
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

### Web Interface

1. Open your browser to `http://localhost:8501`
2. Upload video files
3. Configure processing settings
4. Start processing
5. Monitor progress and view results

## ⚙️ Configuration

### Configuration Files

- `src/config.py`: Main configuration
- `data/settings.json`: User settings (auto-generated)
- `.streamlit/config.toml`: Streamlit theme settings

### Key Settings

#### Processing Settings (`src/config.py`)

```python
processing = ProcessingSettings(
    motion_threshold=3000,      # Motion detection sensitivity
    white_threshold=200,        # White detection threshold
    black_threshold=50,         # Black detection threshold
    use_gpu=True,               # Enable GPU acceleration
    batch_size=4,               # Processing batch size
    worker_processes=2,         # Number of worker processes
)
```

#### Video Output Settings

- **60 Second Video**: Generate 1-minute highlight videos
- **12 Minute Video**: Generate 12-minute compilation videos
- **1 Hour Video**: Generate 1-hour compilation videos
- **Custom Duration**: Specify custom video duration

## 🔍 Troubleshooting

### Common Issues

#### 1. GPU Acceleration Not Working

**Mac (Metal):**

```bash
# Check Metal availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Update PyTorch for better Metal support
pip install --upgrade torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Windows (CUDA):**

```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. FFmpeg Not Found

**Mac:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt install ffmpeg
```

**Windows:**

```powershell
# Download from https://ffmpeg.org/download.html
# Add to system PATH
```

#### 3. Virtual Environment Issues

**Recreate Virtual Environment:**

```bash
# Remove existing
rm -rf nestcam_env

# Create new one
python3.11 -m venv nestcam_env
source nestcam_env/bin/activate  # Mac/Linux
# OR
.\nestcam_env\Scripts\activate.ps1  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 4. Permission Errors

**Windows:**

```powershell
# Run PowerShell as Administrator
# OR use -SkipAdmin parameter
.\install_windows.ps1 -SkipAdmin
```

#### 5. Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall problematic package
pip uninstall package_name
pip install package_name
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
python -m src.main --debug --log-level DEBUG
```

## ⚡ Performance Tips

### GPU Acceleration

1. **Mac**: Metal Performance Shaders automatically enabled
2. **NVIDIA**: Install PyTorch with CUDA support
3. **Fallback**: CPU processing with parallel workers

### Memory Optimization

- Enable "Memory-Efficient Mode" for large files
- Reduce batch size if getting memory errors
- Use streaming processing for 50GB+ files

### Processing Optimization

- Use detailed analysis only when needed
- Adjust motion threshold based on video content
- Use appropriate context window size

### System Resources

- Close unnecessary applications during processing
- Ensure adequate free disk space (10GB+)
- Use SSD storage for better performance

## 🔧 Development

### Project Structure

```
nestcam_processsor_v2/
├── src/
│   ├── main.py                 # Application entry point
│   ├── config.py              # Configuration management
│   ├── ui/
│   │   └── web_app.py         # Streamlit web interface
│   ├── processors/
│   │   ├── video_processor.py # Main video processing
│   │   ├── motion_detector.py # Motion detection algorithms
│   │   └── enhancer.py        # Video enhancement
│   └── services/
│       ├── file_service.py    # File operations
│       ├── analytics_service.py # Analytics
│       └── youtube_service.py # YouTube integration
├── requirements.txt           # Python dependencies
├── install_*.sh/ps1          # Installation scripts
├── activate_venv.*           # Helper scripts
└── README.md                 # This file
```

### Adding New Features

1. **Processing Algorithms**: Add to `src/processors/`
2. **UI Components**: Modify `src/ui/web_app.py`
3. **Services**: Add to `src/services/`
4. **Configuration**: Update `src/config.py`

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest

# Run specific test
pytest tests/test_video_processor.py
```

## 📊 Monitoring

### System Resources

The web interface displays:

- RAM usage and availability
- GPU memory usage (if available)
- Processing progress and statistics
- Real-time logs and debug information

### Performance Metrics

- Processing time per video
- Motion detection accuracy
- Frame processing rate
- Memory usage patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the installation log file
3. Check the [Issues](https://github.com/your-repo/issues) page
4. Create a new issue with:
   - Your OS and version
   - Python version
   - Error messages
   - Installation log

## 📝 Changelog

### Version 2.0.0

- Complete rewrite with GPU acceleration
- Two-pass motion detection
- Cross-platform support
- Web interface with analytics
- Resume functionality
- Virtual environment support

---

**Happy NestCam Processing! 🐦**

```

---

## Summary

**Yes, PowerShell is absolutely the best choice for Windows!** Here's why:

### Why PowerShell is Superior for Windows:

1. **🔧 Native Windows Integration**: Built into Windows, no extra installation needed
2. **🛡️ Security**: Advanced security features and execution policies
3. **📊 Object-Oriented**: Works with .NET objects, much more powerful than batch files
4. **🔍 Excellent Error Handling**: Try-catch blocks and detailed error reporting
5. **📝 Rich Logging**: Built-in transcription and logging capabilities
6. **⚡ Performance**: Faster and more efficient than batch scripts
7. **🔄 Modern Features**: Supports parameters, functions, modules, and more

### Improvements Made:

**Enhanced PowerShell Script:**
- ✅ **Verbose Output**: Detailed step-by-step progress with timestamps
- ✅ **Parameter Support**: `-CleanInstall`, `-Verbose`, `-SkipTests` options
- ✅ **Comprehensive Logging**: Full installation log with timestamps
- ✅ **Error Handling**: Proper try-catch blocks with meaningful error messages
- ✅ **Progress Tracking**: Visual progress indicators and status updates
- ✅ **System Checks**: Verifies Python, GPU, and system compatibility
- ✅ **Helper Scripts**: Easy activation scripts for daily use

**Comprehensive README.md:**
- ✅ **Complete Documentation**: Step-by-step installation for all platforms
- ✅ **Troubleshooting Guide**: Solutions for common issues
- ✅ **Performance Tips**: Optimization recommendations
- ✅ **Usage Examples**: Multiple ways to run the application
- ✅ **System Requirements**: Clear minimum and recommended specs

The PowerShell script now provides enterprise-level installation with full logging, error recovery, and detailed feedback - much better than any batch file could offer!
```
