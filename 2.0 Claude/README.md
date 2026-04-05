# Bird Motion Video Processor

A professional-grade video processing application designed to extract motion from long bird box recordings and create time-lapse videos with adjustable length.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

✅ **Motion Detection** - Intelligently detects and extracts only segments with bird activity  
✅ **Adjustable Time-lapse** - Create 60-second, 10-minute, or 1-hour videos  
✅ **Hardware Acceleration** - Automatic GPU detection (NVIDIA, Intel, Apple)  
✅ **Long Video Support** - Handle videos from 1 to 24+ hours  
✅ **Background Music** - Add music to your time-lapse videos  
✅ **YouTube Upload** - Direct upload to YouTube with privacy settings  
✅ **Modern UI** - Easy-to-use interface with real-time progress tracking  
✅ **Cross-Platform** - Works on Windows, macOS, and Linux  
✅ **Auto-Updates** - Check for updates directly from GitHub  

## Screenshots

[Coming soon]

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **CPU**: Dual-core processor
- **RAM**: 4GB
- **Storage**: 500MB for application + space for videos
- **Python**: 3.8 or higher

### Recommended Requirements
- **CPU**: Quad-core or better
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU (CUDA support) or Intel GPU (Quick Sync) for faster processing
- **Storage**: SSD recommended for better performance

## Installation

### 1. Install System Dependencies

#### Windows
1. **Install Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation

2. **Install FFmpeg**
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to System PATH
   - Verify: Open Command Prompt and run `ffmpeg -version`

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and FFmpeg
brew install python@3.11 ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip ffmpeg
```

### 2. Install Application

```bash
# Clone the repository
git clone https://github.com/yourusername/bird-motion-processor.git
cd bird-motion-processor

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Optional: GPU Acceleration Setup

#### NVIDIA GPU (CUDA)
1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install [cuDNN](https://developer.nvidia.com/cudnn)
3. Install opencv-contrib-python: `pip install opencv-contrib-python`

#### Intel GPU (Quick Sync)
- Windows/Linux: FFmpeg with QSV support (included in most builds)
- No additional setup needed

#### Apple Silicon (M1/M2)
- VideoToolbox acceleration is built into macOS
- No additional setup needed

## Usage

### Starting the Application

```bash
# Make sure virtual environment is activated
python main.py
```

### Basic Workflow

1. **Select Input Video**
   - Click "Browse..." under Input Video
   - Select your bird box recording (MP4, AVI, MOV, MKV)

2. **Configure Output Settings**
   - Choose target length: 60 seconds, 10 minutes, or 1 hour
   - Select output location (or use default)
   - Optionally add background music

3. **Adjust Motion Detection**
   - Use the sensitivity slider (1-10)
   - Lower values = detect more subtle motion
   - Higher values = only detect significant motion

4. **Start Processing**
   - Click "Start Processing"
   - Monitor progress in the log
   - Processing time depends on video length and hardware

5. **View Results**
   - Output video will be saved to selected location
   - Open with any video player

### Advanced Settings

Access the **Settings** tab for fine-tuning:

- **Minimum Motion Duration**: Ignore very brief motion (default: 0.5s)
- **Motion Threshold**: Pixel difference threshold (default: 25)
- **Blur Size**: Noise reduction level (default: 21)
- **GPU Acceleration**: Enable/disable GPU usage
- **CPU Threads**: Number of threads for processing
- **Output Quality**: Low/Medium/High/Maximum

### Adding Background Music

1. Check "Add background music"
2. Click "Browse..." next to music field
3. Select an audio file (MP3, WAV, AAC, M4A)
4. Music will be looped and fade out at the end

### YouTube Upload

1. **Setup (First Time Only)**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable "YouTube Data API v3"
   - Create OAuth 2.0 credentials (Desktop app)
   - Download `client_secrets.json` to app directory

2. **Upload Video**
   - Go to "YouTube Upload" tab
   - Enter title and description
   - Select privacy level
   - Click "Upload to YouTube"
   - Browser will open for authentication (first time)

## Performance Tips

### For Faster Processing

1. **Enable GPU Acceleration**: Settings → Use GPU acceleration
2. **Lower Quality**: Use "Medium" or "Low" quality for drafts
3. **Increase CPU Threads**: Use more CPU cores
4. **Use SSD**: Process videos on SSD rather than HDD
5. **Close Other Apps**: Free up system resources

### For Better Results

1. **Adjust Sensitivity**: 
   - Low (1-3): Detect small movements like head turns
   - Medium (4-7): Normal bird activity
   - High (8-10): Only major movements

2. **Tune Motion Threshold**:
   - Lower: More sensitive (may catch false positives)
   - Higher: Less sensitive (may miss subtle activity)

3. **Adjust Blur Size**:
   - Smaller (11-15): Better detail but more noise
   - Larger (21-31): Better noise reduction but may blur details

## Troubleshooting

### Common Issues

**Issue**: "FFmpeg not found"  
**Solution**: Install FFmpeg and add to PATH. Verify with `ffmpeg -version`

**Issue**: Application won't start  
**Solution**: 
```bash
pip install --upgrade PyQt5
python main.py
```

**Issue**: GPU not detected  
**Solution**: 
- Check GPU drivers are up to date
- For NVIDIA: Install CUDA Toolkit
- Verify in log: "NVIDIA GPU detected" or "Intel Quick Sync detected"

**Issue**: Out of memory  
**Solution**: 
- Reduce number of CPU threads
- Process shorter segments
- Close other applications
- Add more RAM

**Issue**: Motion detection missing activity  
**Solution**: 
- Lower motion sensitivity (slider left)
- Decrease motion threshold in Settings
- Reduce blur size

**Issue**: Too many false positives  
**Solution**: 
- Increase motion sensitivity (slider right)
- Increase minimum motion duration
- Increase blur size

### Log Files

Check `bird_processor.log` for detailed error messages:
```bash
# View last 50 lines
tail -n 50 bird_processor.log  # macOS/Linux
type bird_processor.log | more  # Windows
```

## Command Line Usage (Advanced)

For automation or batch processing, you can use the modules directly:

```python
from motion_detector import MotionDetector
from video_processor import VideoProcessor

config = {
    'sensitivity': 5,
    'min_motion_duration': 0.5,
    'target_length': 60,
    'use_gpu': True
}

# Detect motion
detector = MotionDetector(config)
segments = detector.detect_motion('input.mp4')

# Create time-lapse
processor = VideoProcessor(config)
processor.create_timelapse('input.mp4', segments, 'output.mp4')
```

## Updates

### Checking for Updates

1. Go to "Updates" tab
2. Click "Check for Updates"
3. If update available, click link to download

### Manual Update

```bash
cd bird-motion-processor
git pull origin main
pip install -r requirements.txt --upgrade
```

## Project Structure

```
bird-motion-processor/
├── main.py                 # Main GUI application
├── motion_detector.py      # Motion detection engine
├── video_processor.py      # Video processing and encoding
├── youtube_uploader.py     # YouTube API integration
├── config_manager.py       # Settings management
├── update_checker.py       # GitHub update checker
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── config.json            # User settings (auto-generated)
```

## Configuration File

Settings are stored in `config.json`:

```json
{
    "motion_sensitivity": 5,
    "smoothing": 5,
    "min_motion_duration": 0.5,
    "motion_threshold": 25,
    "blur_size": 21,
    "use_gpu": true,
    "cpu_threads": 7,
    "output_quality": 2
}
```

## Technical Details

### Motion Detection Algorithm

1. **Frame Differencing**: Compares consecutive frames
2. **Gaussian Blur**: Reduces noise and false positives
3. **Thresholding**: Identifies significant changes
4. **Morphological Operations**: Fills gaps in motion regions
5. **Segment Merging**: Combines nearby motion events

### Speed Calculation

```
Speedup Factor = Total Motion Time / Target Length

Example:
- 30 minutes of motion detected
- Target: 60 seconds
- Speedup: 30x (video plays 30 times faster)
```

### Hardware Acceleration

- **NVIDIA**: H.264 NVENC encoder
- **Intel**: Quick Sync Video (QSV)
- **Apple**: VideoToolbox
- **CPU Fallback**: libx264

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Known Limitations

- Maximum video file size depends on available RAM
- GPU acceleration requires compatible drivers
- YouTube upload limited to API quota
- Some video codecs may need transcoding

## Roadmap

- [ ] Multiple camera support
- [ ] Real-time monitoring mode
- [ ] Bird species detection (AI)
- [ ] Cloud processing support
- [ ] Mobile app companion
- [ ] Advanced analytics dashboard

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/bird-motion-processor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bird-motion-processor/discussions)
- **Email**: your.email@example.com

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenCV for computer vision capabilities
- FFmpeg for video processing
- PyQt5 for the user interface
- Google YouTube API for upload functionality

## Version History

### v1.0.0 (2024-01-15)
- Initial release
- Motion detection and time-lapse creation
- Hardware acceleration support
- YouTube upload integration
- Cross-platform support

---

**Made with ❤️ for bird watchers everywhere**
