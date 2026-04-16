@echo off
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Run Setup.bat first.
    pause & exit /b 1
)
call venv\Scripts\activate
python main.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo Check log\processor_log.txt for details.
    pause
)
