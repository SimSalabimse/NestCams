@echo off
REM ================================================================
REM Bird Motion Video Processor v2.0 — Windows Installer
REM Uses OpenCL (standard NVIDIA/AMD/Intel drivers, no CUDA needed)
REM ================================================================

echo.
echo ================================================================
echo Bird Motion Video Processor v2.0 — Installer
echo GPU acceleration via OpenCL (no CUDA required)
echo ================================================================
echo.
pause

REM ── 1. Python ──────────────────────────────────────────────────
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Download Python 3.8+ from https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    pause & exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.8+ required.
    pause & exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo Found Python %%v

REM ── 2. FFmpeg ──────────────────────────────────────────────────
echo.
echo [2/5] Checking FFmpeg...

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo FFmpeg not found in PATH.
    echo.
    choice /C YN /M "Auto-download FFmpeg essentials (~103 MB zip)?"
    if errorlevel 2 (
        echo.
        echo Please install FFmpeg manually:
        echo 1. Go to: https://www.gyan.dev/ffmpeg/builds/
        echo 2. Download "ffmpeg-release-essentials.zip"
        echo 3. Extract it and add the "bin" folder to your system PATH.
        pause & exit /b 1
    )

    echo.
    echo Downloading FFmpeg... (this may take 2-4 minutes - do not close the window)
    
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "[Net.ServicePointManager]::SecurityProtocol = 'Tls12'; " ^
        "try { " ^
            "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' " ^
            "-OutFile 'ffmpeg.zip' -UseBasicParsing -ErrorAction Stop; " ^
            "Write-Host 'Download finished successfully.' " ^
        "} catch { " ^
            "Write-Host 'Download ERROR:' $_.Exception.Message; " ^
            "exit 1 " ^
        "}"

    if errorlevel 1 (
        echo.
        echo ERROR: Download failed. Please check your internet connection.
        pause & exit /b 1
    )

    echo.
    echo Extracting FFmpeg... (this can take 1-2 minutes)
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "try { " ^
            "Expand-Archive -Path ffmpeg.zip -DestinationPath . -Force -ErrorAction Stop; " ^
            "Write-Host 'Extraction finished successfully.' " ^
        "} catch { " ^
            "Write-Host 'Extraction ERROR:' $_.Exception.Message; " ^
            "exit 1 " ^
        "}"

    del ffmpeg.zip 2>nul

    REM Find the extracted folder (usually starts with "ffmpeg-")
    for /d %%d in (ffmpeg-*) do set "FFDIR=%%d"

    if not defined FFDIR (
        echo ERROR: Could not find the extracted FFmpeg folder.
        pause & exit /b 1
    )

    set "PATH=%CD%\%FFDIR%\bin;%PATH%"
    echo ✓ FFmpeg successfully extracted to: %FFDIR%
    echo.
    echo IMPORTANT: Add this folder permanently to your system PATH:
    echo %CD%\%FFDIR%\bin
)

REM Final verification
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ FFmpeg is ready
) else (
    echo ERROR: FFmpeg still not detected.
    echo Please add the "bin" folder to your system PATH and run the installer again.
    pause & exit /b 1
)

REM ── 3. Virtual environment ─────────────────────────────────────
echo.
echo [3/5] Setting up virtual environment...
if not exist venv (
    python -m venv venv
    echo ✓ Created virtual environment
) else (
    echo Using existing virtual environment
)
call venv\Scripts\activate.bat

REM ── 4. Install Python packages ─────────────────────────────────
echo.
echo [4/5] Installing Python packages...

echo Removing conflicting OpenCV packages...
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python >nul 2>&1

echo Installing opencv-contrib-python (with OpenCL support)...
pip install "opencv-contrib-python>=4.8.0"
if errorlevel 1 (
    echo ERROR: Failed to install OpenCV.
    pause & exit /b 1
)

echo Installing remaining packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt
    pause & exit /b 1
)

REM ── 5. Verify OpenCL ───────────────────────────────────────────
echo.
echo [5/5] Checking OpenCL GPU acceleration...
python -c "import cv2; cv2.ocl.setUseOpenCL(True); ok = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL(); print(' OpenCL available:', ok); print(' Devices:', cv2.ocl.getOpenCLDevices())" 2>nul

echo.
echo ================================================================
echo Installation completed successfully!
echo ================================================================
echo.
echo Next steps:
echo   • Test GPU acceleration:   python test_opencl.py
echo   • Start the application:    START.bat   or   python main.py
echo.
pause