# First Time User Guide

Welcome! This guide will walk you through your first video processing session step-by-step.

## What You'll Learn
1. How to install and set up the application
2. How to process your first bird box video
3. How to adjust settings for better results
4. How to troubleshoot common issues

**Time Required**: 15-30 minutes (including first video processing)

---

## Part 1: Installation (10 minutes)

### Step 1: Install System Requirements

#### Windows Users

**1.1 Install Python**
- Download from [python.org](https://www.python.org/downloads/)
- Run installer
- ✅ **IMPORTANT**: Check "Add Python to PATH"
- Click "Install Now"
- Verify: Open Command Prompt, type `python --version`

**1.2 Install FFmpeg**
- Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Extract ZIP to `C:\ffmpeg`
- Add to PATH:
  1. Right-click "This PC" → Properties
  2. Advanced System Settings → Environment Variables
  3. Under "System variables", find "Path", click Edit
  4. Click "New", add `C:\ffmpeg\bin`
  5. Click OK on all windows
- Verify: Open NEW Command Prompt, type `ffmpeg -version`

#### macOS Users

**1.1 Install Homebrew** (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**1.2 Install Python and FFmpeg**
```bash
brew install python@3.11 ffmpeg
```

**1.3 Verify**
```bash
python3 --version
ffmpeg -version
```

#### Linux Users (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg
```

### Step 2: Get the Application

**2.1 Clone Repository**
```bash
git clone https://github.com/yourusername/bird-motion-processor.git
cd bird-motion-processor
```

OR download ZIP from GitHub and extract

### Step 3: Install Python Dependencies

**3.1 Create Virtual Environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3.2 Run Setup Script**
```bash
python setup.py
```

This will:
- Check your system
- Install all required packages
- Detect GPU capabilities
- Create necessary directories

**3.3 Verify Installation**
```bash
python test_installation.py
```

You should see all tests pass ✅

---

## Part 2: First Video Processing (10-20 minutes)

### Step 1: Prepare Your Test Video

**For First Time**: Use a SHORT test video (1-5 minutes)
- Easier to debug if something goes wrong
- Faster processing
- Verify everything works

**Good Test Videos**:
- 2-3 minutes of bird box footage
- Some motion, some static periods
- Normal lighting conditions

### Step 2: Launch Application

```bash
# Make sure virtual environment is active
python main.py
```

You should see the application window open.

### Step 3: Select Input Video

1. Click **"Browse..."** button under "Input Video"
2. Navigate to your test video
3. Select it and click Open
4. Path should appear in the label

### Step 4: Configure Basic Settings

**Target Length** (dropdown):
- Choose **"60 seconds"** for your first test
- This makes a 1-minute time-lapse

**Motion Sensitivity** (slider):
- Leave at **5 (Medium)** for first test
- You can adjust this later

**Output Location**:
- Leave as default (same folder as input)
- Or click "Choose..." to pick different location

### Step 5: Start Processing

1. Click the big **"Start Processing"** button
2. Watch the progress bar
3. Monitor the log output

**What You'll See**:
```
Starting processing...
Analyzing video for motion... 0%
Detecting motion: 25.3%
Detecting motion: 50.1%
...
Processing video segments...
Creating time-lapse: 75.2%
...
Video saved to: [path]
```

**Processing Time Estimates**:
- 2 min video → 30-60 seconds
- 5 min video → 1-2 minutes
- 1 hr video → 3-8 minutes (varies by hardware)

### Step 6: View Your Result

1. Processing completes with success message
2. Note the output file path
3. Navigate to that folder
4. Open the video in any video player
5. Watch your time-lapse!

---

## Part 3: Understanding Your Results

### What to Look For

**Good Result**:
- ✅ Video plays smoothly
- ✅ Shows all bird activity
- ✅ Minimal empty/static frames
- ✅ Good video quality

**Common Issues**:
- ❌ Too much static footage → Increase sensitivity
- ❌ Missing bird activity → Decrease sensitivity
- ❌ Jumpy/choppy video → Adjust speed smoothing
- ❌ Poor quality → Use higher quality preset

### Motion Sensitivity Guide

The sensitivity slider controls what counts as "motion":

**Low (1-3)**: Very sensitive
- Catches tiny movements
- May include camera shake, shadows
- **Use for**: Night/IR cameras, distant birds, subtle movements

**Medium (4-7)**: Balanced (DEFAULT)
- Catches normal bird activity
- Filters most false positives
- **Use for**: Typical day footage, active birds

**High (8-10)**: Less sensitive
- Only major movements
- May miss small birds or brief visits
- **Use for**: Very active birds, windy conditions

### Reading the Log

Important log messages:

```
"Detected 15 motion segments"
→ Good! Found activity in video

"Total motion time: 28.3s out of 180.0s (15.7%)"
→ 28 seconds of motion in 3-minute video
→ Seems reasonable for bird box

"Speedup factor: 0.47x"
→ WARNING: Motion is longer than target
→ Video won't be sped up
```

---

## Part 4: Adjusting Settings

### When Results Aren't Perfect

#### Problem: Too Much Static Footage

**Solution**: Increase sensitivity
- Move slider RIGHT (towards 8-10)
- OR go to Settings → Increase "Motion Threshold"
- Try again

#### Problem: Missing Bird Activity

**Solution**: Decrease sensitivity
- Move slider LEFT (towards 1-3)
- OR go to Settings → Decrease "Motion Threshold"
- OR Settings → Decrease "Minimum Motion Duration"

#### Problem: Video Too Short/Long

The app targets your chosen length (60s, 10min, etc.) by:
1. Finding total motion time
2. Calculating speedup needed
3. Speeding up video

**If motion < target**: Video won't be sped up, stays natural speed

**If you want exact length**: You need enough motion in original video

**Example**:
- Want: 60s output
- Have: 30s of motion detected
- Result: Video will be 30s (can't slow down to reach 60s)

#### Problem: Poor Video Quality

**Solution**: Increase quality
1. Go to **Settings** tab
2. Change "Output Quality" to "High" or "Maximum"
3. Process again

Note: Higher quality = longer processing time

### Advanced Settings (Settings Tab)

**Minimum Motion Duration**:
- Default: 0.5 seconds
- Increase to ignore very brief movements
- Decrease to catch fleeting visits

**Motion Threshold**:
- Default: 25 (pixel difference)
- Increase for less sensitive detection
- Decrease for more sensitive detection

**Blur Size**:
- Default: 21
- Larger = more noise filtering (smoother)
- Smaller = more detail (may catch noise)

**CPU Threads**:
- Default: Max cores - 1
- Increase to use more CPU (faster)
- Decrease if computer feels slow

---

## Part 5: Processing Long Videos (1+ hours)

### Special Considerations

**Memory**: Long videos use more RAM
- Close other applications
- Consider lowering quality for first pass
- Monitor system resources

**Time**: Be patient!
- 1 hour video: 5-15 minutes processing
- 6 hour video: 15-45 minutes
- 24 hour video: 30-90 minutes

**Strategy**:
1. Test with short clip first (verify settings)
2. Process full video overnight
3. Check results in morning

### Recommended Settings for Long Videos

```
Motion Sensitivity: 6-7 (slightly higher)
Output Quality: Medium (for draft, High for final)
CPU Threads: Max (use all cores)
GPU Acceleration: ON (if available)
```

---

## Part 6: Adding Background Music

### Step 1: Prepare Music

- Format: MP3, WAV, AAC, or M4A
- Length: Any (will be looped/trimmed automatically)
- Recommendation: Calm, nature sounds

### Step 2: Add to Video

1. Check **"Add background music"**
2. Click **"Browse..."** next to music field
3. Select your audio file
4. Process video normally

**Result**: Music will:
- Loop if needed
- Fade out in last 2 seconds
- Match video length exactly

---

## Part 7: Common First-Time Issues

### Issue: "FFmpeg not found"

**Cause**: FFmpeg not installed or not in PATH

**Fix**:
1. Verify installation: `ffmpeg -version` in terminal
2. If not found: Install FFmpeg (see Step 1)
3. Make sure to restart terminal/app after adding to PATH

### Issue: "No motion detected"

**Possible Causes**:
- Video is actually static (no bird visits)
- Sensitivity too low
- Video is corrupted

**Fix**:
1. Watch original video - does it have motion?
2. Lower sensitivity slider to 1-2
3. Settings → Lower motion threshold to 10-15
4. Try different video

### Issue: Application won't start

**Fix**:
```bash
pip install --upgrade PyQt5
pip install --upgrade opencv-python
python main.py
```

### Issue: "Out of memory"

**Cause**: Video too large for available RAM

**Fix**:
1. Close other applications
2. Settings → Reduce CPU threads
3. Use lower quality setting
4. Process shorter segments

### Issue: Very slow processing

**Possible Causes**:
- No GPU acceleration
- Too many CPU threads
- High quality settings

**Check**:
1. Look at log: Does it detect GPU?
2. Settings → Try "Low" or "Medium" quality
3. Reduce CPU threads slightly

---

## Part 8: Next Steps

### You're Ready to Process Real Videos!

**Workflow**:
1. Record bird box footage (any length)
2. Copy to computer
3. Launch Bird Motion Processor
4. Select video
5. Choose target length
6. Adjust sensitivity if needed
7. Process!
8. Share with friends/family

### Advanced Features to Explore

**YouTube Upload**:
- See README.md for API setup
- Upload directly from app

**Custom Settings**:
- Save your favorite configurations
- Load for consistent results

**Batch Processing**:
- Process multiple videos
- Use saved settings for consistency

### Tips for Best Results

1. **Test First**: Always test settings on short clip
2. **Adjust Gradually**: Change one setting at a time
3. **Take Notes**: Record what works for your setup
4. **Save Settings**: Save when you find good configuration
5. **Check Logs**: Logs help diagnose issues

---

## Quick Reference

### Perfect First-Time Settings

```
Target Length: 60 seconds
Motion Sensitivity: 5 (Medium)
Speed Smoothing: 5
Output Quality: Medium (for test), High (for keeper)
GPU Acceleration: ON
```

### When to Adjust

| Observation | Adjustment |
|------------|------------|
| Too much static video | Sensitivity higher (→ right) |
| Missing bird activity | Sensitivity lower (← left) |
| Video choppy/jumpy | Increase speed smoothing |
| Poor quality | Higher quality preset |
| Slow processing | Lower quality, fewer threads |
| Very brief visits missing | Lower min motion duration |

---

## Getting Help

**Check First**:
1. This guide
2. README.md (comprehensive)
3. QUICKSTART.md (quick overview)
4. Log file: `bird_processor.log`

**Still Stuck?**:
- GitHub Issues
- Include: Log file, error message, video specs
- Be specific about problem

---

## Success Checklist

✅ Application installed and launches  
✅ Test video processes successfully  
✅ Output video plays correctly  
✅ Understand how to adjust sensitivity  
✅ Know how to read logs  
✅ Ready to process real bird box footage  

**Congratulations!** You're now ready to create amazing bird box time-lapses! 🎉🐦

---

## What's Next?

- Process your longest bird box recording
- Experiment with different music
- Share videos on YouTube
- Fine-tune settings for your specific setup
- Contribute improvements (see CONTRIBUTING.md)

Happy bird watching! 🐦
