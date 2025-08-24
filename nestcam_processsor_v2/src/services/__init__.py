"""
Services package for NestCam Processor
"""

# Optional imports with fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not available - memory monitoring disabled")

import gc

try:
    import tempfile
    import shutil
except ImportError:
    print("⚠️ tempfile/shutil not available - some features may not work")

# GPU Support (CUDA for 4070ti)
try:
    import cupy as cp
    HAS_GPU = True
    print("✅ GPU acceleration enabled (CUDA)")
except ImportError:
    HAS_GPU = False
    print("⚠️ GPU acceleration not available (install cupy for CUDA support)")


# Memory monitoring (optional)
def get_memory_usage():
    """Get current memory usage in GB"""
    if HAS_PSUTIL:
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except:
            return 0.0
    else:
        return 0.0  # Return 0 if psutil not available


def log_memory_usage(message):
    """Log current memory usage"""
    mem_gb = get_memory_usage()
    if mem_gb > 0:
        print(f"{message} - Memory: {mem_gb:.2f}GB")
    else:
        print(f"{message} - Memory monitoring not available")
