@echo off
echo.
echo ================================================================
echo  Bird Box Video Processor v12.0 — Windows Setup
echo ================================================================
echo.

python --version >nul 2>&1 || (
    echo ERROR: Python not found.
    echo Download Python 3.9+ from https://www.python.org/downloads/
    echo Check "Add Python to PATH" during installation!
    pause & exit /b 1
)

where ffmpeg >nul 2>&1 || (
    echo ERROR: FFmpeg not found in PATH.
    echo Download from https://www.gyan.dev/ffmpeg/builds/
    echo Extract and add the bin\ folder to your system PATH.
    pause & exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate

echo Removing conflicting OpenCV builds...
pip uninstall -y opencv-python opencv-python-headless 2>nul

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: Installation failed. & pause & exit /b 1 )

echo.
echo Checking OpenCL GPU acceleration...
python -c "import cv2; cv2.ocl.setUseOpenCL(True); ok=cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL(); print('  OpenCL:', ok)"

echo.
echo ================================================================
echo  Setup complete! Run Start.bat to launch.
echo ================================================================
pause
