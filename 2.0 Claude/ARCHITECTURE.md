# Architecture Documentation

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface (PyQt5)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Process │  │ Settings │  │  Upload  │  │  Updates │   │
│  │   Tab    │  │   Tab    │  │   Tab    │  │   Tab    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Application Logic Layer                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              ProcessingThread (QThread)               │  │
│  │  - Runs in background                                │  │
│  │  - Keeps UI responsive                               │  │
│  │  - Emits progress signals                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Core Processing Modules                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Motion     │  │    Video     │  │   Config     │     │
│  │  Detector    │  │  Processor   │  │   Manager    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │   YouTube    │  │   Update     │                       │
│  │  Uploader    │  │   Checker    │                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    External Dependencies                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   OpenCV     │  │    FFmpeg    │  │  Google API  │     │
│  │  (Computer   │  │   (Video     │  │  (YouTube    │     │
│  │   Vision)    │  │  Encoding)   │  │   Upload)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. main.py - User Interface Layer

**Responsibilities:**
- Render GUI using PyQt5
- Handle user interactions
- Manage UI state
- Display progress and logs
- Spawn background processing threads

**Key Classes:**
- `MainWindow`: Main application window
- `ProcessingThread`: Background worker thread

**Design Patterns:**
- **MVC Pattern**: Separates UI from business logic
- **Observer Pattern**: Progress callbacks and signals
- **Singleton**: Single application instance

### 2. motion_detector.py - Motion Detection Engine

**Responsibilities:**
- Analyze video frames for motion
- Apply computer vision algorithms
- Identify time segments with activity
- Filter noise and false positives

**Algorithm Flow:**
```
1. Load video → Get frame properties
2. For each frame:
   a. Convert to grayscale
   b. Apply Gaussian blur (noise reduction)
   c. Compute difference from previous frame
   d. Threshold to binary (motion/no-motion)
   e. Morphological operations (fill gaps)
   f. Count motion pixels
   g. Mark frame if motion detected
3. Convert motion frames to time segments
4. Merge nearby segments
5. Return filtered segment list
```

**Key Classes:**
- `MotionDetector`: CPU-based detection
- `GPUMotionDetector`: GPU-accelerated (future)

**Optimizations:**
- Gaussian blur for noise reduction
- Configurable sensitivity
- Smart segment merging
- Minimum duration filtering

### 3. video_processor.py - Video Processing Engine

**Responsibilities:**
- Extract motion segments from source video
- Apply speed adjustments
- Concatenate segments
- Add background music
- Encode final output

**Processing Pipeline:**
```
Input Video + Motion Segments
    ↓
Extract Segment → FFmpeg (-ss, -t)
    ↓
Speed Up → setpts filter (PTS adjustment)
    ↓
Encode → Hardware encoder (NVENC/QSV/VideoToolbox)
    ↓
Repeat for all segments
    ↓
Concatenate → FFmpeg concat demuxer
    ↓
Add Music (optional) → Audio mix + fade
    ↓
Final Encode → Quality settings applied
    ↓
Output Video
```

**Hardware Acceleration:**
- **NVIDIA**: H.264 NVENC encoder
- **Intel**: Quick Sync Video (QSV)
- **Apple**: VideoToolbox
- **Fallback**: libx264 (CPU)

**Key Methods:**
- `_detect_hardware_acceleration()`: Auto-detect GPU
- `_extract_segment()`: Extract and speed up
- `_concatenate_segments()`: Merge segments
- `_add_music()`: Background music integration
- `_encode_final()`: Final quality encoding

### 4. config_manager.py - Configuration System

**Responsibilities:**
- Load/save user settings
- Provide default configuration
- Persist settings across sessions
- Validate configuration values

**Configuration Storage:**
- Format: JSON
- Location: `config.json` (same directory)
- Auto-generated on first run
- Settings survive app restarts

**Design Pattern:**
- **Singleton-like**: Single config file
- **Lazy Loading**: Load when needed
- **Fail-safe**: Defaults if file missing/corrupt

### 5. youtube_uploader.py - YouTube Integration

**Responsibilities:**
- Authenticate with Google OAuth2
- Upload videos to YouTube
- Set metadata (title, description, privacy)
- Handle API errors gracefully

**Authentication Flow:**
```
First Time:
1. Check for client_secrets.json
2. Launch OAuth flow in browser
3. User grants permissions
4. Save credentials to youtube_token.json

Subsequent Uploads:
1. Load saved credentials
2. Refresh if expired
3. Upload directly
```

**API Usage:**
- YouTube Data API v3
- Resumable upload for large files
- Progress callbacks for UI updates

### 6. update_checker.py - Update System

**Responsibilities:**
- Check GitHub for new releases
- Compare version numbers
- Provide download links
- Handle network errors

**Update Check Flow:**
```
1. Request GitHub API: /repos/{owner}/{repo}/releases/latest
2. Parse JSON response
3. Extract version tag (e.g., "v1.2.0")
4. Compare with current version
5. Return update status + download URL
```

**Version Comparison:**
- Semantic versioning (MAJOR.MINOR.PATCH)
- Tuple comparison (1.2.0 → (1, 2, 0))
- Handles missing/malformed versions

## Data Flow

### Complete Processing Flow

```
User Clicks "Start Processing"
    ↓
Validate inputs (file exists, output path valid)
    ↓
Create config dictionary from UI settings
    ↓
Spawn ProcessingThread (background)
    ↓
┌─────────────────────────────────────┐
│     In Background Thread:           │
│                                     │
│  1. MotionDetector.detect_motion()  │
│     - Load video                    │
│     - Process frames                │
│     - Identify motion segments      │
│     - Return [(start, end), ...]    │
│                                     │
│  2. VideoProcessor.create_timelapse()│
│     - Calculate speedup factor      │
│     - Extract each segment          │
│     - Concatenate segments          │
│     - Add music (if requested)      │
│     - Final encoding                │
│     - Return success/failure        │
│                                     │
│  3. Emit completion signal          │
└─────────────────────────────────────┘
    ↓
UI receives signal
    ↓
Display success message + output path
    ↓
Enable "Start Processing" button
```

### Progress Updates

```
Processing Thread          UI Thread
      │                       │
      │──progress_signal──────>│ Update progress bar
      │                       │ Update status label
      │──progress_signal──────>│ Append to log
      │                       │
      │──complete_signal──────>│ Show message box
      │                       │ Re-enable button
```

## Error Handling Strategy

### Layered Error Handling

```
Level 1: UI Layer
- Validate inputs before processing
- Show user-friendly error dialogs
- Keep app running despite errors

Level 2: Processing Layer
- Try-catch around major operations
- Log exceptions with full traceback
- Return error status to UI

Level 3: Module Layer
- Validate parameters
- Handle edge cases gracefully
- Provide meaningful error messages

Level 4: External Dependencies
- Check tool availability (FFmpeg)
- Handle subprocess failures
- Timeout protection for long operations
```

### Error Recovery

```
Input Validation Failed
    → Show error dialog
    → Don't start processing
    → Log issue

Motion Detection Failed
    → Log error
    → Return empty segment list
    → UI shows "No motion detected"

Video Processing Failed
    → Cleanup temp files
    → Log exception
    → UI shows error details

External Tool Failed (FFmpeg)
    → Capture stderr
    → Log command + output
    → Show helpful error message
```

## Threading Model

### Main Thread (UI)
- Handles all UI events
- Never blocks on I/O
- Updates display from signals

### Background Thread
- Video processing (I/O intensive)
- Motion detection (CPU intensive)
- Emits progress signals
- Cleans up on completion/error

### Thread Safety
- No shared mutable state
- Communication via Qt signals (thread-safe)
- Config read-only during processing

## Performance Optimizations

### 1. Hardware Detection
```python
# Automatic optimization based on available hardware
if NVIDIA_GPU_detected:
    use_nvenc_encoder()
elif Intel_GPU_detected:
    use_qsv_encoder()
elif Apple_Silicon:
    use_videotoolbox()
else:
    use_cpu_multithread()
```

### 2. Processing Strategy
```
Short Videos (<1hr):
- Direct processing
- High quality settings
- More CPU threads

Long Videos (>6hr):
- Batch processing
- Balanced quality
- Memory-conscious
```

### 3. Memory Management
```
- Stream processing (not load entire video)
- Temporary file cleanup
- Segment-by-segment approach
- Configurable buffer sizes
```

## Extensibility Points

### Adding New Features

**1. Custom Motion Detectors**
```python
class AIMotionDetector(MotionDetector):
    """ML-based bird detection"""
    def detect_motion(self, video_path):
        # Custom implementation
        pass
```

**2. Export Formats**
```python
class VideoProcessor:
    def export_gif(self, ...):
        # Add GIF export
        pass
```

**3. Cloud Integration**
```python
class CloudUploader:
    """Upload to Dropbox/Drive/S3"""
    def upload(self, ...):
        pass
```

## Configuration Philosophy

### Default Values
- **Conservative**: Won't overwhelm weak systems
- **Balanced**: Good results without manual tuning
- **Overridable**: All settings exposed in UI

### Setting Persistence
- Save on explicit user action (Save button)
- Don't auto-save during processing
- Preserve settings across updates

## Security Considerations

### 1. File Path Handling
- Validate all file paths
- No arbitrary code execution
- Sanitize file names

### 2. External Process Execution
- Fixed command structure
- No shell injection
- Timeout protection

### 3. API Credentials
- Never commit secrets to git (.gitignore)
- OAuth token storage (encrypted by Google library)
- Local file storage only

### 4. Network Requests
- HTTPS only
- Timeout on all requests
- Handle connection failures

## Testing Strategy

### Unit Testing (Future)
```python
# Motion detection
test_empty_video()
test_static_video()
test_continuous_motion()

# Video processing
test_speedup_calculation()
test_segment_extraction()
test_concatenation()

# Config
test_save_load()
test_defaults()
```

### Integration Testing
```
Test full pipeline:
1. Load sample video
2. Detect motion
3. Create time-lapse
4. Verify output plays
```

### Performance Testing
```
Benchmark on:
- 1 hour video (CPU)
- 1 hour video (GPU)
- 24 hour video
- Various resolutions
```

## Deployment Architecture

### Desktop Application
```
Distribution:
- Source code (Git)
- Installer packages (Future):
  - Windows: .exe installer
  - macOS: .dmg disk image
  - Linux: .deb/.rpm packages
```

### Dependencies Management
```
Runtime:
- Python 3.8+
- FFmpeg (system-installed)

Python Packages:
- requirements.txt (pip install)
- Virtual environment recommended
```

## Future Architecture Improvements

### 1. Plugin System
```python
class PluginInterface:
    def on_motion_detected(self, segments):
        pass
    
    def on_processing_complete(self, output_path):
        pass
```

### 2. Web Interface
```
FastAPI backend
React frontend
WebSocket for real-time updates
```

### 3. Distributed Processing
```
Master node: Coordinates
Worker nodes: Process segments
Result aggregation
```

### 4. Database Integration
```
SQLite for:
- Processing history
- Statistics tracking
- Favorite settings
```

## Summary

The architecture is designed to be:
- **Modular**: Each component has clear responsibility
- **Extensible**: Easy to add new features
- **Performant**: Hardware optimization + threading
- **Reliable**: Error handling at every level
- **Maintainable**: Clear separation of concerns
- **User-friendly**: Simple interface, complex internals

The layered approach ensures the UI remains responsive while heavy processing happens in the background, and the modular design allows for easy testing and future enhancements.
