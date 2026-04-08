#!/usr/bin/env python3
"""
OpenCL Diagnostic Script
Run this to check if GPU acceleration is working.
"""

import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("  Bird Motion Processor — OpenCL Diagnostics")
    print("=" * 60)

    # 1. OpenCV import
    try:
        import cv2
        print(f"\n✅ OpenCV {cv2.__version__} installed")
    except ImportError:
        print("\n❌ OpenCV not installed — run: pip install opencv-contrib-python")
        sys.exit(1)

    # 2. OpenCL availability
    print(f"\n--- OpenCL Status ---")
    have = cv2.ocl.haveOpenCL()
    use = cv2.ocl.useOpenCL()
    print(f"haveOpenCL(): {have}")
    print(f"useOpenCL():  {use}")

    if not have:
        print("\n❌ OpenCL not available in this OpenCV build.")
        print("   Fix: pip install opencv-contrib-python")
        print("   Also: make sure GPU drivers are up to date.")
        sys.exit(1)

    if not use:
        print("\n⚠️  OpenCL available but not enabled. Enabling now…")
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            print("   ✅ OpenCL enabled successfully")
        else:
            print("   ❌ Could not enable OpenCL")

    # 3. Force NVIDIA device
    os.environ["OPENCV_OPENCL_DEVICE"] = "NVIDIA:GPU"
    cv2.ocl.setUseOpenCL(True)

    # 4. Device info
    try:
        device = cv2.ocl.Device.getDefault()
        if device.available():
            print(f"\n✅ OpenCL device: {device.name()}")
            print(f"   Vendor:  {device.vendorName()}")
            mem_mb = device.globalMemSize() / 1024 / 1024
            print(f"   Memory:  {mem_mb:.0f} MB")
            print(f"   OpenCL version: {device.OpenCLVersion()}")
        else:
            print("\n⚠️  No OpenCL device available — will use CPU")
    except Exception as e:
        print(f"\n⚠️  Could not get device info: {e}")

    # 5. Quick benchmark
    print("\n--- Speed Test (UMat vs CPU) ---")
    import numpy as np
    import time

    SIZE = (1080, 1920, 3)
    test_frame = np.random.randint(0, 255, SIZE, dtype=np.uint8)

    # CPU
    N = 20
    t0 = time.time()
    for _ in range(N):
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    cpu_time = (time.time() - t0) / N * 1000

    # GPU (UMat)
    umat = cv2.UMat(test_frame)
    # Warmup
    for _ in range(3):
        g = cv2.cvtColor(umat, cv2.COLOR_BGR2GRAY)
    t0 = time.time()
    for _ in range(N):
        g = cv2.cvtColor(umat, cv2.COLOR_BGR2GRAY)
        b = cv2.GaussianBlur(g, (7, 7), 0)
        _, th = cv2.threshold(b, 25, 255, cv2.THRESH_BINARY)
        _ = cv2.countNonZero(th)  # force execution + scalar download
    gpu_time = (time.time() - t0) / N * 1000

    print(f"CPU:  {cpu_time:.1f} ms/frame")
    print(f"GPU:  {gpu_time:.1f} ms/frame")
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"✅ GPU is {speedup:.1f}x faster — OpenCL acceleration working!")
    else:
        print("⚠️  GPU not faster than CPU — OpenCL may not be targeting your dGPU.")
        print("   Try setting: OPENCV_OPENCL_DEVICE=NVIDIA:GPU  in your environment.")

    # 6. FFmpeg check
    print("\n--- FFmpeg Encoders ---")
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        for enc in ["h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "h264_qsv", "libx264"]:
            mark = "✅" if enc in r.stdout else "  "
            print(f"  {mark} {enc}")
    except FileNotFoundError:
        print("❌ FFmpeg not found — install from ffmpeg.org")

    print("\n" + "=" * 60)
    print("  Diagnostics complete")
    print("=" * 60)


if __name__ == "__main__":
    main()