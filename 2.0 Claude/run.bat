@echo off
REM Bird Motion Video Processor - Windows Launcher

echo Starting Bird Motion Video Processor...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
    echo.
)

REM Run the application
python main.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo Check bird_processor.log for details.
    pause
)
