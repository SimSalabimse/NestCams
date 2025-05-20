#!/bin/bash

echo "Updating repository..."
# Adjust path to match Raspberry Pi's filesystem (e.g., /home/pi/NestCams)
cd /home/simsalabim/Documents/NestCams || exit
git pull

echo "Starting..."
# Activate the virtual environment
source venv/bin/activate

echo "Started"
# Run the Python script (assuming movie_7.0.2.pyw is in the current directory)
python3 movie_8.2.0.pyw