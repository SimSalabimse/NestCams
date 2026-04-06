@echo off
REM ===============================================================
REM Bird Motion Video Processor - Automatic Installer for Windows
REM Handles Python, FFmpeg, CUDA OpenCV, and all dependencies
REM ===============================================================

echo.
echo ================================================================
echo    Bird Motion Video Processor v2.0 - Automatic Installer
echo ================================================================
echo.
echo This installer will set up everything you need:
echo   - Python virtual environment
echo   - FFmpeg with NVENC support
echo   - CUDA-enabled OpenCV (for GPU acceleration)
echo   - All Python dependencies
echo.
pause

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo WARNING: Not running as administrator.
    echo Some features may require admin rights.
    echo Right-click this file and select "Run as administrator" for best results.
    echo.
    pause
)

REM ==================== CHECK PYTHON ====================
echo.
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Check Python version is 3.8+
python -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.8 or higher is required!
    echo You have Python %PYTHON_VERSION%
    echo Please install a newer version from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✓ Python version OK

REM ==================== CHECK/INSTALL FFMPEG ====================
echo.
echo [2/6] Checking FFmpeg installation...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo FFmpeg not found in PATH!
    echo.
    choice /C YN /M "Do you want to auto-download FFmpeg?"
    if errorlevel 2 goto MANUAL_FFMPEG
    
    echo Downloading FFmpeg...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg.zip'}"
    
    echo Extracting FFmpeg...
    powershell -Command "Expand-Archive -Path ffmpeg.zip -DestinationPath . -Force"
    
    REM Find the extracted folder
    for /d %%i in (ffmpeg-*) do set FFMPEG_DIR=%%i
    
    REM Add to PATH for this session
    set "PATH=%CD%\%FFMPEG_DIR%\bin;%PATH%"
    
    echo.
    echo FFmpeg downloaded to: %CD%\%FFMPEG_DIR%
    echo.
    echo IMPORTANT: To use FFmpeg permanently, add this to your system PATH:
    echo %CD%\%FFMPEG_DIR%\bin
    echo.
    echo Or run this as administrator to add automatically:
    powershell -Command "Write-Host 'setx PATH \"%PATH%;%CD%\%FFMPEG_DIR%\bin\" /M' -ForegroundColor Yellow"
    echo.
    pause
    
    del ffmpeg.zip
    goto FFMPEG_DONE
    
    :MANUAL_FFMPEG
    echo.
    echo Please download FFmpeg manually:
    echo 1. Go to: https://www.gyan.dev/ffmpeg/builds/
    echo 2. Download: ffmpeg-release-essentials.zip
    echo 3. Extract to C:\ffmpeg
    echo 4. Add C:\ffmpeg\bin to your System PATH
    echo.
    echo Then run this installer again.
    pause
    exit /b 1
)

:FFMPEG_DONE
echo ✓ FFmpeg found

REM Check for NVENC support
ffmpeg -hide_banner -encoders 2>nul | findstr /C:"h264_nvenc" >nul
if errorlevel 1 (
    echo.
    echo WARNING: NVENC encoder not detected in FFmpeg
    echo You may not get GPU-accelerated encoding.
    echo.
) else (
    echo ✓ NVENC encoder detected
)

REM ==================== CREATE VIRTUAL ENVIRONMENT ====================
echo.
echo [3/6] Creating Python virtual environment...
if exist venv (
    echo Virtual environment already exists, using existing one...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo ✓ Virtual environment activated

REM ==================== CHECK GPU ====================
echo.
echo [4/6] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected - will use CPU mode
    echo.
    echo If you have an NVIDIA GPU:
    echo   1. Install latest drivers from https://www.nvidia.com/drivers
    echo   2. Run this installer again
    echo.
    set GPU_MODE=cpu
    pause
) else (
    echo ✓ NVIDIA GPU detected
    nvidia-smi --query-gpu=name --format=csv,noheader
    set GPU_MODE=gpu
)

REM ==================== INSTALL DEPENDENCIES ====================
echo.
echo [5/6] Installing Python dependencies...
echo This may take several minutes...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

REM Remove any existing OpenCV installations
echo Removing old OpenCV installations...
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >nul 2>&1

if "%GPU_MODE%"=="gpu" (
    echo.
    echo Installing CUDA-enabled OpenCV for GPU acceleration...
    echo This is a large download and may take 5-10 minutes...
    echo.
    
    REM Try to install opencv-python-cuda
    pip install opencv-python-cuda==4.10.0.84
    if errorlevel 1 (
        echo.
        echo WARNING: Failed to install opencv-python-cuda
        echo Falling back to CPU-only OpenCV...
        pip install opencv-python>=4.8.0
    ) else (
        echo ✓ CUDA OpenCV installed successfully
    )
) else (
    echo Installing CPU-only OpenCV...
    pip install opencv-python>=4.8.0
)

echo.
echo Installing other dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some dependencies!
    echo Check the error messages above.
    pause
    exit /b 1
)
echo ✓ All dependencies installed

REM ==================== VERIFY INSTALLATION ====================
echo.
echo [6/6] Verifying installation...
echo.

python -c "import cv2; print('✓ OpenCV version:', cv2.__version__)" 2>nul
if errorlevel 1 (
    echo ✗ OpenCV import failed!
    set INSTALL_OK=0
) else (
    set INSTALL_OK=1
)

python -c "import PyQt5; print('✓ PyQt5 installed')" 2>nul
if errorlevel 1 (
    echo ✗ PyQt5 import failed!
    set INSTALL_OK=0
)

python -c "import numpy; print('✓ NumPy version:', numpy.__version__)" 2>nul
if errorlevel 1 (
    echo ✗ NumPy import failed!
    set INSTALL_OK=0
)

REM Check CUDA if GPU mode
if "%GPU_MODE%"=="gpu" (
    python -c "import cv2; count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0; print(f'✓ CUDA devices: {count}'); exit(0 if count > 0 else 1)" 2>nul
    if errorlevel 1 (
        echo.
        echo WARNING: CUDA not available in OpenCV
        echo You'll use CPU mode. For GPU acceleration:
        echo   1. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
        echo   2. Reinstall with: pip install opencv-python-cuda==4.10.0.84
        echo.
    ) else (
        echo ✓ GPU acceleration ready!
    )
)

echo.
if %INSTALL_OK%==1 (
    echo ================================================================
    echo    Installation Complete! 🎉
    echo ================================================================
    echo.
    echo To start the application:
    echo   1. Double-click: START.bat
    echo   2. Or run: python main_gpu.py
    echo.
    echo For help and documentation:
    echo   - README.md - Complete guide
    echo   - QUICKSTART.md - 5-minute start
    echo   - GPU_OPTIMIZATION_GUIDE.md - GPU setup
    echo.
    echo Repository: https://github.com/SimSalabimse/NestCams
    echo Branch: 2.0 / Folder: 2.0 Claude
    echo.
) else (
    echo ================================================================
    echo    Installation completed with WARNINGS
    echo ================================================================
    echo.
    echo Some components failed to install.
    echo Check the messages above and try:
    echo   1. Updating Python: python -m pip install --upgrade pip
    echo   2. Running as administrator
    echo   3. Checking internet connection
    echo.
)

pause
