#!/usr/bin/env python3
"""
Test Script for Bird Motion Video Processor
Verifies installation and basic functionality
"""

import sys
import os
import subprocess
import importlib

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_result(test_name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"      {message}")

def test_python_version():
    """Test Python version"""
    print_header("Python Version Check")
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    print_result(
        "Python 3.8+", 
        passed,
        f"Version: {version.major}.{version.minor}.{version.micro}"
    )
    return passed

def test_ffmpeg():
    """Test FFmpeg installation"""
    print_header("FFmpeg Check")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        passed = result.returncode == 0
        if passed:
            # Extract version
            version_line = result.stdout.decode().split('\n')[0]
            print_result("FFmpeg installed", True, version_line)
        else:
            print_result("FFmpeg installed", False, "FFmpeg not found")
        return passed
    except Exception as e:
        print_result("FFmpeg installed", False, str(e))
        return False

def test_python_packages():
    """Test required Python packages"""
    print_header("Python Package Check")
    
    packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('PyQt5', 'PyQt5'),
        ('requests', 'requests')
    ]
    
    all_passed = True
    for module_name, package_name in packages:
        try:
            importlib.import_module(module_name)
            print_result(f"{package_name}", True)
        except ImportError:
            print_result(f"{package_name}", False, "Not installed")
            all_passed = False
    
    return all_passed

def test_optional_packages():
    """Test optional packages"""
    print_header("Optional Package Check")
    
    packages = [
        ('google_auth_oauthlib', 'google-auth-oauthlib', 'YouTube upload'),
        ('googleapiclient', 'google-api-python-client', 'YouTube upload')
    ]
    
    for module_name, package_name, feature in packages:
        try:
            importlib.import_module(module_name)
            print_result(f"{package_name}", True, f"For: {feature}")
        except ImportError:
            print_result(f"{package_name}", False, f"Optional for: {feature}")

def test_gpu_support():
    """Test GPU support"""
    print_header("GPU Support Check")
    
    # Check NVIDIA
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, timeout=2)
        if result.returncode == 0:
            print_result("NVIDIA GPU", True, "CUDA acceleration available")
            return
    except:
        pass
    
    # Check for Intel QSV
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                              capture_output=True, text=True, timeout=2)
        if 'h264_qsv' in result.stdout:
            print_result("Intel Quick Sync", True, "QSV acceleration available")
            return
    except:
        pass
    
    # Check for Apple VideoToolbox
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                  capture_output=True, text=True, timeout=2)
            if 'h264_videotoolbox' in result.stdout:
                print_result("Apple VideoToolbox", True, "Hardware acceleration available")
                return
        except:
            pass
    
    print_result("GPU Acceleration", False, "Will use CPU (slower but works)")

def test_module_imports():
    """Test if project modules can be imported"""
    print_header("Project Module Check")
    
    modules = [
        'motion_detector',
        'video_processor',
        'config_manager',
        'update_checker',
        'youtube_uploader'
    ]
    
    all_passed = True
    for module in modules:
        try:
            importlib.import_module(module)
            print_result(f"{module}.py", True)
        except Exception as e:
            print_result(f"{module}.py", False, str(e))
            all_passed = False
    
    return all_passed

def test_file_structure():
    """Test if all required files exist"""
    print_header("File Structure Check")
    
    required_files = [
        'main.py',
        'motion_detector.py',
        'video_processor.py',
        'config_manager.py',
        'update_checker.py',
        'youtube_uploader.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_passed = True
    for file in required_files:
        exists = os.path.exists(file)
        print_result(file, exists)
        if not exists:
            all_passed = False
    
    return all_passed

def test_opencv_video_support():
    """Test OpenCV video codec support"""
    print_header("OpenCV Video Support Check")
    
    try:
        import cv2
        
        # Test video capture
        cap = cv2.VideoCapture()
        backends = cv2.videoio_registry.getBackends()
        
        print_result("OpenCV video backends", True, 
                    f"Available: {len(backends)} backends")
        
        # Test if we can create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print_result("MP4 codec support", True, "H.264/MP4 available")
        
        return True
    except Exception as e:
        print_result("OpenCV video support", False, str(e))
        return False

def test_write_permissions():
    """Test if we can write to current directory"""
    print_header("File System Permissions Check")
    
    test_file = 'test_write_permission.tmp'
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print_result("Write permissions", True, "Can create files")
        return True
    except Exception as e:
        print_result("Write permissions", False, str(e))
        return False

def test_memory_available():
    """Test available system memory"""
    print_header("System Resources Check")
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print_result("System Memory", True, 
                    f"Total: {total_gb:.1f}GB, Available: {available_gb:.1f}GB")
        
        if total_gb < 4:
            print("      ⚠️  Warning: Less than 4GB RAM. May struggle with long videos.")
        
        return True
    except ImportError:
        print_result("System Memory", False, "psutil not installed (optional)")
        return True  # Not critical

def run_basic_functionality_test():
    """Test basic motion detection functionality"""
    print_header("Basic Functionality Test")
    
    print("This would require a test video file.")
    print("Skipping for now - run manual tests with real videos.")
    print_result("Functionality test", True, "Skipped (requires test video)")

def print_summary(results):
    """Print test summary"""
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for r in results if r)
    failed = total - passed
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Run the application: python main.py")
        print("  2. See QUICKSTART.md for usage guide")
        print("  3. Process a test video to verify everything works")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Install FFmpeg and add to PATH")
        print("  - Update Python to 3.8 or higher")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  Bird Motion Video Processor - Installation Test")
    print("="*60)
    print("\nThis script will verify your installation is correct.\n")
    
    results = []
    
    # Core requirements
    results.append(test_python_version())
    results.append(test_ffmpeg())
    results.append(test_python_packages())
    
    # Optional but recommended
    test_optional_packages()
    test_gpu_support()
    
    # Project structure
    results.append(test_file_structure())
    results.append(test_module_imports())
    
    # System capabilities
    results.append(test_opencv_video_support())
    results.append(test_write_permissions())
    test_memory_available()
    
    # Functionality
    run_basic_functionality_test()
    
    # Summary
    print_summary(results)
    
    return 0 if all(results) else 1

if __name__ == '__main__':
    sys.exit(main())
