#!/usr/bin/env python3
"""
Setup script for Bird Motion Video Processor
Helps with initial setup and dependency checking
"""

import sys
import subprocess
import platform
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg installed")
            return True
    except:
        pass
    
    print("❌ FFmpeg not found")
    print("\nTo install FFmpeg:")
    
    system = platform.system()
    if system == "Windows":
        print("  1. Download from https://ffmpeg.org/download.html")
        print("  2. Extract to C:\\ffmpeg")
        print("  3. Add C:\\ffmpeg\\bin to System PATH")
    elif system == "Darwin":  # macOS
        print("  brew install ffmpeg")
    else:  # Linux
        print("  sudo apt install ffmpeg  # Ubuntu/Debian")
        print("  sudo yum install ffmpeg  # CentOS/RHEL")
    
    return False

def install_requirements():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 
                       'requirements.txt'], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_gpu():
    """Check for GPU acceleration support"""
    print("\n🎮 Checking for GPU acceleration...")
    
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print("   Install CUDA Toolkit for GPU acceleration:")
            print("   https://developer.nvidia.com/cuda-downloads")
            return
    except:
        pass
    
    # Check for Intel GPU
    system = platform.system()
    if system in ['Windows', 'Linux']:
        print("ℹ️  Intel Quick Sync may be available (check FFmpeg encoders)")
    elif system == 'Darwin':
        print("✅ Apple VideoToolbox available on macOS")
    else:
        print("ℹ️  No GPU detected, will use CPU processing")

def create_directories():
    """Create necessary directories"""
    dirs = ['logs', 'temp']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"✅ Created directory: {d}")

def main():
    print("="*60)
    print("Bird Motion Video Processor - Setup")
    print("="*60)
    
    print("\n🔍 Checking system requirements...\n")
    
    checks_passed = True
    
    # Check Python
    if not check_python_version():
        checks_passed = False
    
    # Check FFmpeg
    if not check_ffmpeg():
        checks_passed = False
    
    if not checks_passed:
        print("\n⚠️  Some requirements are missing. Please install them first.")
        sys.exit(1)
    
    # Install dependencies
    print("\n" + "="*60)
    if not install_requirements():
        print("\n⚠️  Failed to install dependencies")
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Create directories
    print("\n" + "="*60)
    create_directories()
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("\nTo run the application:")
    print("  python main.py")
    print("\nFor YouTube upload, see README for API setup instructions")
    print("="*60)

if __name__ == '__main__':
    main()
