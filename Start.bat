@echo off
echo Updating repository...
cd /d "%~dp0"
git pull

echo Starting...
call venv\Scripts\activate
echo Started
python movie_8.2.2.pyw
