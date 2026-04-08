# Bird Motion Video Processor v2.0

Extracts motion from long bird box recordings and creates time-lapse videos.  
GPU-accelerated via **OpenCL** (standard NVIDIA/AMD/Intel drivers — no CUDA required).

---

## Features

- **OpenCL motion detection** — works on RTX cards with standard Game Ready drivers
- **Hardware encoding** — NVENC (NVIDIA), VideoToolbox (Apple), QSV (Intel), libx264 fallback
- 60s vertical / 10min / 1hr time-lapses in one batch pass
- Background music with smooth looping and fade-out
- Frame quality filtering (black, white, corrupted frames)
- Dark mode UI with real-time progress and ETA
- Real-time folder monitoring for automated processing
- YouTube upload
- Analytics dashboard

---

## Quick Start

### 1. Install System Requirements

**Windows:**
```
Python 3.8+  — python.org (check "Add to PATH")
FFmpeg       — gyan.dev/ffmpeg/builds → extract → add bin/ to PATH
```

**macOS:**
```bash
brew install python@3.11 ffmpeg
```

**Linux:**
```bash
sudo apt install python3 python3-pip ffmpeg
```

### 2. Install Python Packages

```bash
# Remove any old OpenCV first
pip uninstall opencv-python opencv-python-headless -y

# Install (includes OpenCL support)
pip install -r requirements.txt
```

### 3. Verify GPU Acceleration

```bash
python test_opencl.py
```

You should see your GPU listed and a speedup vs CPU.

### 4. Run

```bash
python main.py
# or on Windows: double-click START.bat
```

---

## Usage

1. **Browse** → select your bird box recording (MP4, MOV, AVI, MKV)
2. Choose target length: 60s, 10min, 1hr, or custom
3. Optionally enable batch mode to produce all three lengths at once
4. Adjust **Motion Sensitivity** (1 = very sensitive, 10 = only large movements)
5. Click **Start Processing**
6. Output video(s) saved to the same folder as input (or your chosen location)

---

## Settings Reference

| Setting | Default | Description |
|---|---|---|
| Sensitivity | 5 | 1–10; lower detects subtler motion |
| Min motion duration | 0.5 s | Ignore very brief events |
| Segment padding | 1.0 s | Extra context before/after each clip |
| Frame skip | 2 | Process every Nth frame (higher = faster) |
| Output quality | High | Low / Medium / High / Maximum |

---

## GPU Acceleration

See **OPENCL_GUIDE.md** for full details.

**tl;dr:**
```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
python test_opencl.py
```

---

## YouTube Upload

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → enable YouTube Data API v3
3. Create OAuth 2.0 credentials (Desktop app) → download `client_secrets.json`
4. Place `client_secrets.json` in the app folder
5. Use the YouTube Upload tab — browser will open for first-time auth

---

## Troubleshooting

**"No motion detected"** → lower the sensitivity slider or decrease min motion duration

**"FFmpeg not found"** → install FFmpeg and add its `bin/` folder to your PATH

**OpenCL not working** → update GPU drivers, then run `python test_opencl.py`

**tmix / FFmpeg filter error** → already fixed in this version; the `weights` format issue from older builds is resolved

**Output file has no extension** → fixed; `.mp4` is now appended automatically if missing

---

## File Structure

```
├── main.py                 # GUI application
├── motion_detector.py      # OpenCL-accelerated motion detection
├── video_processor.py      # FFmpeg encoding pipeline
├── config_manager.py       # Settings persistence
├── update_checker.py       # GitHub update checker
├── youtube_uploader.py     # YouTube API upload
├── analytics_dashboard.py  # SQLite statistics
├── real_time_monitor.py    # Folder watch / auto-process
├── dark_mode.py            # Qt stylesheet
├── bird_detector.py        # Optional AI bird detection
├── test_opencl.py          # GPU diagnostics
├── requirements.txt
├── OPENCL_GUIDE.md         # GPU setup guide
├── INSTALL.bat             # Windows installer
└── START.bat               # Windows launcher
```

---

## License

MIT — see LICENSE