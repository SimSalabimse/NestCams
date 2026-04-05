#!/bin/bash
# Bird Motion Video Processor - Unix Launcher

echo "Starting Bird Motion Video Processor..."
echo ""

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Using system Python."
    echo ""
fi

# Run the application
python3 main.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with an error."
    echo "Check bird_processor.log for details."
    read -p "Press Enter to continue..."
fi
