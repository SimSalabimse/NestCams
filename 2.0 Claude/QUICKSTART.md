# Quick Start Guide

Get up and running with Bird Motion Video Processor in 5 minutes!

## Prerequisites

✅ Windows 10/11, macOS 10.15+, or Linux  
✅ Python 3.8 or higher  
✅ FFmpeg installed  

## Installation (3 Steps)

### Step 1: Get the Code
```bash
git clone https://github.com/yourusername/bird-motion-processor.git
cd bird-motion-processor
```

### Step 2: Run Setup
```bash
python setup.py
```

This will:
- Check system requirements
- Install all Python dependencies
- Detect GPU capabilities
- Create necessary directories

### Step 3: Launch
```bash
python main.py
```

## First Time Use

### Basic Processing

1. **Click "Browse..."** under Input Video
2. **Select your bird box recording**
3. **Choose target length**: 60 seconds, 10 minutes, or 1 hour
4. **Click "Start Processing"**
5. **Wait for completion** (time varies by video length)
6. **Find your video** in the same folder as input

### Typical Processing Times

| Input Length | Output  | Approximate Time* |
|--------------|---------|------------------|
| 1 hour       | 60s     | 2-5 minutes      |
| 6 hours      | 10min   | 10-20 minutes    |
| 24 hours     | 1hr     | 30-60 minutes    |

*Times vary based on hardware and settings

## Quick Tips

### Better Motion Detection
- **Daytime videos**: Sensitivity 5-7
- **Night/IR videos**: Sensitivity 3-5  
- **Active birds**: Sensitivity 7-9

### Faster Processing
1. Enable GPU acceleration (Settings tab)
2. Use "Medium" quality for drafts
3. Process on SSD if available

### Best Quality
1. Use "High" or "Maximum" quality
2. Increase CPU threads to max
3. Adjust motion threshold in Settings

## Common First-Time Issues

**"FFmpeg not found"**  
→ Install FFmpeg and add to PATH  
→ Windows: Download from ffmpeg.org  
→ Mac: `brew install ffmpeg`  
→ Linux: `sudo apt install ffmpeg`

**"No motion detected"**  
→ Lower sensitivity slider (move left)  
→ Check video actually has motion  
→ Try Settings → Lower motion threshold

**App crashes on start**  
→ Run: `pip install --upgrade PyQt5`  
→ Check Python version: `python --version`  

## Next Steps

📖 Read the full [README.md](README.md) for:
- Advanced settings
- YouTube upload setup
- Performance optimization
- Troubleshooting

🎥 Try different sensitivity settings to find what works for your setup

⚙️ Explore the Settings tab for fine-tuning

## Need Help?

- Check [README.md](README.md) for detailed docs
- View logs in `bird_processor.log`
- Report issues on GitHub

---

Happy bird watching! 🐦
