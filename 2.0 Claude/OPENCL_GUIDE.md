# OpenCL GPU Acceleration Guide

## Why OpenCL Instead of CUDA?

OpenCL works through **standard GPU drivers** — no special CUDA build of OpenCV required. Your RTX 4070 Ti is fully supported out of the box via NVIDIA's Game Ready drivers.

**Performance vs CPU:**

| Hardware | 24-hour video | Approximate time |
|---|---|---|
| CPU only (8 cores) | 24 hours | 60–90 min |
| RTX 4070 Ti (OpenCL) | 24 hours | ~10–20 min |

Motion detection is the main beneficiary. Video encoding (NVENC) was already GPU-accelerated.

---

## Setup (3 commands)

```bash
# 1. Remove any conflicting OpenCV
pip uninstall opencv-python opencv-python-headless -y

# 2. Install OpenCV with OpenCL support
pip install opencv-contrib-python

# 3. Verify
python test_opencl.py
```

That's it. No CUDA Toolkit, no special builds.

---

## Verifying GPU Is Active

Run the built-in test:

```bash
python test_opencl.py
```

Expected output:
```
✅ OpenCV 4.x.x installed
haveOpenCL(): True
useOpenCL():  True
✅ OpenCL device: NVIDIA GeForce RTX 4070 Ti
   Vendor:  NVIDIA Corporation
   Memory:  12288 MB
GPU:  X.X ms/frame
CPU:  Y.Y ms/frame
✅ GPU is N.Nx faster — OpenCL acceleration working!
```

---

## If OpenCL Shows as Not Available

**Step 1** — Update your NVIDIA driver from GeForce Experience or [nvidia.com/drivers](https://www.nvidia.com/drivers).

**Step 2** — If you have both an Intel iGPU and NVIDIA dGPU, force NVIDIA:
```python
import os
os.environ["OPENCV_OPENCL_DEVICE"] = "NVIDIA:GPU"
```
This is already set automatically in `motion_detector.py`.

**Step 3** — Reinstall OpenCV:
```bash
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-contrib-python
```

**Step 4** — Confirm the check:
```python
import cv2
print(cv2.ocl.haveOpenCL())   # must be True
print(cv2.ocl.useOpenCL())    # must be True
```

---

## How It Works

The `MotionDetector` uses OpenCV's `UMat` API:

```python
# All heavy operations run on GPU via UMat
umat = cv2.UMat(frame)                            # upload frame
cv2.accumulateWeighted(umat, avg_umat, 0.02)      # GPU
diff = cv2.absdiff(umat, avg_umat)                # GPU
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)     # GPU
blurred = cv2.GaussianBlur(gray, (7,7), 0)        # GPU
_, thresh = cv2.threshold(blurred, 25, 255, ...)  # GPU
dilated = cv2.dilate(thresh, None, iterations=3)  # GPU
motion_pixels = cv2.countNonZero(dilated)         # single scalar download
```

Only one integer is transferred from GPU to CPU per frame — this is what makes it fast.

---

## Tuning

| Setting | Default | Effect |
|---|---|---|
| `sensitivity` | 5 | Lower = more sensitive to motion |
| `frame_skip` | 2 | Higher = faster but less accurate |
| `segment_padding` | 1.0 s | Extra context before/after motion |
| `detection_scale` | 640 px | Lower = faster detection |
| `bg_learn_rate` | 0.02 | Lower = more stable background model |

For 4K source videos with lots of false positives (wind/leaves), try:
- `sensitivity` = 7–8
- `frame_skip` = 4
- `detection_scale` = 480

---

## AMD and Intel GPUs

OpenCL also works on AMD (Adrenalin drivers) and Intel Arc/UHD:

- AMD: remove `OPENCV_OPENCL_DEVICE=NVIDIA:GPU` from environment, or set it to `AMD:GPU`
- Intel: set to `Intel:GPU`
- Auto (pick whatever's available): remove the env var entirely

---

## NVENC Still Used for Encoding

OpenCL handles **motion detection**. Video **encoding** still uses NVENC (hardware H.264 encoder), which was already working. Both work independently.