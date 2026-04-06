@echo off
REM ===============================================================
REM Bird Motion Video Processor - Quick Start Launcher
REM ===============================================================

echo Starting Bird Motion Video Processor v2.0...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo.
    echo Please run INSTALL.bat first to set up the application.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if activation worked
where python | findstr venv >nul
if errorlevel 1 (
    echo Failed to activate virtual environment!
    echo Please run INSTALL.bat to fix the installation.
    pause
    exit /b 1
)

REM Launch application
python main_gpu.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo Check bird_processor.log for details.
    echo.
    pause
)
