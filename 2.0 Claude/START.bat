@echo off
REM Bird Motion Video Processor — Quick Launcher

echo Starting Bird Motion Video Processor v2.0...
echo.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found — run INSTALL.bat first.
    pause & exit /b 1
)

python main.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo Check bird_processor.log for details.
    pause
)