# NestCams 2.0 - Video Processor for Motion Detection and Time-Lapse

A cross-platform application to detect motion in videos, extract motion segments, create time-lapses, add music, and upload to YouTube.

## Features

- Motion detection using OpenCV
- Time-lapse creation with adjustable lengths (60s, 10min, 1hr)
- Music overlay
- YouTube upload integration
- Hardware optimization (GPU/CPU detection)
- Modern PyQt5 UI
- Update checking from GitHub
- Cross-platform support (Windows, macOS, Linux)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/NestCams.git
   cd NestCams
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. For YouTube upload, set up Google API:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a project and enable YouTube Data API v3
   - Create OAuth 2.0 credentials
   - Download client_secrets.json and place it in the project root

## Usage

Run the application:

```bash
python -m ui.main_window
```

Or after installation:

```bash
nestcams
```

## System Requirements

- Python 3.8+
- FFmpeg installed on system
- For GPU acceleration: CUDA-compatible GPU (optional)

## Troubleshooting

- If OpenCV fails to install, install system dependencies:
  - Windows: Download FFmpeg binaries and add to PATH
  - macOS: `brew install ffmpeg opencv`
  - Linux: `sudo apt install ffmpeg libopencv-dev`

- For YouTube upload issues, ensure client_secrets.json is correctly configured

## License

MIT License
