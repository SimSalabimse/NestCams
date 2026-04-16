# Pipeline Order & Why It Is Optimal

## Stage Overview

```
Input video (24 h)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — Motion Detection  (motion_detector.py)          │
│  • Downsample frames to 640 px width                        │
│  • Skip over/under-exposed frames                           │
│  • Apply optional ROI mask                                  │
│  • Adjacent-frame diff on GPU (OpenCL UMat)                 │
│  • Exposure-step correction (IR-cut / auto-exposure)        │
│  • Spatial-uniformity flicker rejection (std/mean ratio)    │
│  • Contour size filter (MIN_CONTOUR_AREA)                   │
│  • Optional MOG2 background-subtractor secondary check      │
│  • Temporal consistency gate (N-of-M window)                │
│  → List of (start_sec, end_sec) segments                    │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 — Segment Extraction  (video_processor.py)        │
│  • FFmpeg stream-copy per segment (-avoid_negative_ts)      │
│  • No re-encoding — lossless and fast                       │
│  → N individual .mp4 files in a temp directory              │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 — Concatenation  (video_processor.py)             │
│  • FFmpeg concat demuxer (-reset_timestamps 1)              │
│  → Single concat.mp4 (still lossless, timestamps clean)     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 — Post-process + Final Encode  (ONE FFmpeg pass)   │
│  (video_processor.py + filters.py)                          │
│                                                             │
│  VF chain (in this exact order):                            │
│   1. setpts=PTS-STARTPTS         reset timestamp drift      │
│   2. deflicker=size=5            per-frame brightness spikes│
│   3. eq=contrast:saturation:...  global colour grade        │
│   4. hqdn3d                      spatial+temporal denoise   │
│   5. tmix=frames=N               motion blur  ← BEFORE spd │
│   6. setpts=PTS/SPEED            speedup      ← AFTER blur  │
│   7. transpose=1                 rotate (Shorts only)       │
│   8. drawtext=...                watermark (optional)       │
│                                                             │
│  AF chain:                                                  │
│   atempo chain (chained stages for speed > 2×)             │
│   afade=t=out (last 2 s)                                    │
│   amix with music track (if provided)                       │
│                                                             │
│  Encode:  libx264 / h264_nvenc                              │
│  Hard cap: -t TARGET_LENGTH (never exceeds target)          │
│  → Final output .mp4                                        │
└─────────────────────────────────────────────────────────────┘
```

## Why this order is optimal

### Detection first
Reduces the data the pipeline needs to process from 24 hours to typically
5–15 minutes.  All expensive filters only run on the motion-only clip.

### Stream-copy extraction + concat
Zero quality loss.  These two stages are I/O-bound; no CPU/GPU time is
spent on pixel operations.

### Everything after concat in one FFmpeg pass
Re-encoding degrades quality (generation loss) and wastes time.  By fusing
stages 3 and 4 into a single command, the pixel data flows from the input
through the filter graph and directly into the encoder without any
intermediate file write at full quality.

### deflicker before speedup
`deflicker` corrects per-frame brightness by comparing adjacent frames.
After speedup those "adjacent frames" already represent multiple original
frames — the temporal comparison is meaningless and the filter has no effect.

### tmix (motion blur) before setpts (speedup)
`tmix` blends N consecutive source frames into one output frame, simulating
a longer physical camera shutter.  If applied after `setpts`, the frames
being blended are already "virtual" (resampled outputs) — the effect
disappears.  Applied before speedup, each blended frame corresponds to real
source content and the result looks genuinely cinematic.

### Rotation last (for Shorts)
`transpose=1` is a geometric transform.  By placing it after all the colour
and temporal filters, those filters process the smaller original-orientation
frame, saving CPU/GPU cycles.
