#!/bin/bash
# Helper script to activate the NestCam virtual environment

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/nestcam_env"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the installation script first:"
    echo "   ./install_mac.sh  (on Mac)"
    echo "   ./install_linux.sh  (on Linux)"
    exit 1
fi

echo "🔗 Activating NestCam virtual environment..."
source "$VENV_DIR/bin/activate"

if [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    echo ""
    echo "🚀 To run the application:"
    echo "   python -m src.main --web"
    echo ""
    echo "📝 To deactivate later: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
