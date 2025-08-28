#!/bin/bash
# NestCam Processor v3.0 - Enhanced Launcher

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/nestcam_env"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first: python setup.py"
    exit 1
fi

echo "🔗 Activating NestCam virtual environment..."
source "$VENV_DIR/bin/activate"

if [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    echo ""
    echo "🚀 Available commands:"
    echo "   python -m src.main --web          # Start web interface"
    echo "   python -m src.main --cli          # Start CLI mode"
    echo "   python test_installation.py       # Test installation"
    echo "   python setup.py --help           # Setup help"
    echo ""
    echo "📝 To deactivate: deactivate"
    
    # If arguments provided, run the application
    if [ $# -gt 0 ]; then
        python -m src.main "$@"
    fi
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
