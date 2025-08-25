#!/usr/bin/env python3
"""
Test script to verify NestCam Processor installation
"""

import sys
import importlib
from pathlib import Path


def test_import(module_name: str) -> bool:
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False


def test_gpu_acceleration():
    """Test GPU acceleration"""
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (Metal) available")
            device = torch.device("mps")
            test_tensor = torch.randn(10, 10, device=device)
            print("✅ Metal GPU test passed")
        else:
            print("⚠️ No GPU acceleration available")
        return True
    except ImportError:
        print("❌ PyTorch not available")
        return False


def main():
    """Run all installation tests"""
    print("🧪 NestCam Processor Installation Test")
    print("=" * 40)

    # Test core dependencies
    core_modules = [
        "streamlit",
        "cv2",
        "numpy",
        "pandas",
        "plotly",
        "psutil",
        "pydantic",
    ]

    print("\n📚 Testing core dependencies:")
    core_results = [test_import(module) for module in core_modules]

    print("\n🎯 Testing GPU acceleration:")
    gpu_result = test_gpu_acceleration()

    print("\n📁 Testing project imports:")
    project_modules = [
        "src.config",
        "src.ui.web_app",
        "src.processors.video_processor",
        "src.services.analytics_service",
    ]

    # Add src to path for testing
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    project_results = []
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
            project_results.append(True)
        except ImportError as e:
            print(f"❌ {module}: {e}")
            project_results.append(False)

    # Summary
    print("\n" + "=" * 40)
    all_passed = all(core_results + [gpu_result] + project_results)

    if all_passed:
        print("🎉 All tests passed! Installation successful.")
        print("\n🚀 To run the application:")
        print("   python -m src.main --web")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n🔧 To fix issues:")
        print("   1. Reinstall missing dependencies: pip install -r requirements.txt")
        print("   2. Check GPU drivers if GPU tests failed")
        print("   3. Ensure you're in the virtual environment")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
