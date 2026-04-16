#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[ -d .git ] && git pull --quiet
source venv/bin/activate
python3 main.py
