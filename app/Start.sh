#!/bin/bash

echo "Updating repository..."
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || exit
git pull

echo "Starting..."
source venv/bin/activate

echo "Started"
python main.pyw