@echo off
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
echo Setup complete. Use Start.bat to run the application.
pause