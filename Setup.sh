#!/bin/bash

# Check if virtual environment exists, create it if it doesn't
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Display completion message
echo "Setup complete. Use start.sh to run the application."

# Pause to mimic Windows behavior (press any key to continue)
read -p "Press Enter to continue..."