@echo off
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements_beta.txt
echo Setup complete. Use Start.bat to run the application.
pause
