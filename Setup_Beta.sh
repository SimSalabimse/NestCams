# #!/bin/bash

# # Check if virtual environment exists, create it if it doesn't
# if [ ! -d "venv" ]; then
#     python3 -m venv venv
# fi

# # Activate the virtual environment
# source venv/bin/activate

# # Install dependencies from requirements.txt
# pip install -r requirements_beta.txt

# # Display completion message
# echo "Setup complete. Use Start.sh to run the application."

# # Pause to mimic Windows behavior (press any key to continue)
# read -p "Press Enter to continue..."

#!/bin/bash
# Setup script for Video Processor

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
MIN_VERSION="3.8"
if [[ "$(printf '%s\n' "$PYTHON_VERSION" "$MIN_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]]; then
    echo "Error: Python $PYTHON_VERSION found, but Python 3.8 or later is required."
    exit 1
fi

# Create and activate virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install dependencies
pip install --upgrade pip
if ! pip install -r requirements_beta.txt; then
    echo "Error: Failed to install dependencies. Check requirements_beta.txt."
    deactivate
    exit 1
fi

echo "Setup complete. Use Start_beta.sh to run the application."
read -p "Press Enter to continue..."