@echo off
echo Updating repository...
cd D:\Sander\Videos\DayRecordings2025\NestCams
git pull

echo Starting...
call D:\Sander\Videos\DayRecordings2025\NestCams\venv\Scripts\activate
echo Started
python D:\Sander\Videos\DayRecordings2025\NestCams\movie_7.0.2.pyw

