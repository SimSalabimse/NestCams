#!/usr/bin/env python3
"""
Enhanced NestCam Processor Setup Script
Automatically detects platform and installs optimized dependencies
"""

import sys
import os
import platform
import subprocess
import argparse
from pathlib import Path
import json
import shutil


class NestCamSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.system = platform.system().lower()
        self.arch = platform.machine().lower()

        # Platform-specific configurations
        self.platform_configs = {
            "darwin": {
                "name": "macOS",
                "package_manager": "brew",
                "python_cmd": "python3.11",
                "gpu_support": ["metal", "cpu"],
                "recommended_python": "3.11",
            },
            "linux": {
                "name": "Linux",
                "package_manager": "apt",
                "python_cmd": "python3.11",
                "gpu_support": ["cuda", "cpu"],
                "recommended_python": "3.11",
            },
            "windows": {
                "name": "Windows",
                "package_manager": "choco",
                "python_cmd": "python",
                "gpu_support": ["cuda", "cpu"],
                "recommended_python": "3.11",
            },
        }

    def run(self):
        """Main setup process"""
        print("🐦 NestCam Processor v3.0 - Enhanced Setup")
        print("=" * 50)

        # Parse command line arguments
        parser = argparse.ArgumentParser(description="NestCam Processor Setup")
        parser.add_argument(
            "--gpu",
            choices=["auto", "cuda", "metal", "cpu"],
            default="auto",
            help="GPU acceleration mode",
        )
        parser.add_argument(
            "--minimal", action="store_true", help="Minimal installation (CPU only)"
        )
        parser.add_argument(
            "--dev",
            action="store_true",
            help="Development installation with extra tools",
        )
        parser.add_argument(
            "--clean",
            action="store_true",
            help="Clean reinstall (remove existing venv)",
        )

        args = parser.parse_args()

        # Detect and configure platform
        config = self.detect_platform()

        # Apply user preferences
        if args.gpu != "auto":
            config["gpu_mode"] = args.gpu
        if args.minimal:
            config["minimal"] = True
        if args.dev:
            config["dev_mode"] = True
        if args.clean:
            config["clean_install"] = True

        # Run installation
        self.install_platform_dependencies(config)
        self.create_optimized_venv(config)
        self.install_python_dependencies(config)
        self.configure_gpu_acceleration(config)
        self.setup_project_config(config)
        self.create_launcher_scripts(config)
        self.run_post_install_tests(config)

        self.show_completion_message(config)

    def detect_platform(self):
        """Detect platform and return configuration"""
        if self.system not in self.platform_configs:
            print(f"❌ Unsupported platform: {self.system}")
            sys.exit(1)

        config = self.platform_configs[self.system].copy()
        config.update(
            {
                "system": self.system,
                "arch": self.arch,
                "gpu_mode": "auto",
                "minimal": False,
                "dev_mode": False,
                "clean_install": False,
            }
        )

        print(f"✅ Detected: {config['name']} ({self.arch})")
        return config

    def install_platform_dependencies(self, config):
        """Install platform-specific system dependencies"""
        print(f"\n📦 Installing {config['name']} system dependencies...")

        if config["system"] == "darwin":
            self._install_macos_deps(config)
        elif config["system"] == "linux":
            self._install_linux_deps(config)
        elif config["system"] == "windows":
            self._install_windows_deps(config)

    def _install_macos_deps(self, config):
        """Install macOS dependencies"""
        deps = ["ffmpeg"]
        if not config["minimal"]:
            deps.extend(["libomp", "tesseract"])

        for dep in deps:
            try:
                result = subprocess.run(
                    ["brew", "list", dep], capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"📦 Installing {dep}...")
                    subprocess.run(["brew", "install", dep], check=True)
                else:
                    print(f"✅ {dep} already installed")
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to install {dep}, continuing...")

    def _install_linux_deps(self, config):
        """Install Linux dependencies"""
        deps = ["ffmpeg", "python3-dev", "build-essential"]
        if not config["minimal"]:
            deps.extend(["libtesseract-dev", "tesseract-ocr"])

        try:
            subprocess.run(["sudo", "apt", "update"], check=True)
            for dep in deps:
                print(f"📦 Installing {dep}...")
                subprocess.run(["sudo", "apt", "install", "-y", dep], check=True)
        except subprocess.CalledProcessError:
            print("⚠️ Some system dependencies may not have installed")

    def _install_windows_deps(self, config):
        """Install Windows dependencies"""
        # Check for Chocolatey
        try:
            subprocess.run(["choco", "--version"], capture_output=True, check=True)
            has_choco = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            has_choco = False

        if has_choco:
            deps = ["ffmpeg"]
            if not config["minimal"]:
                deps.append("tesseract")

            for dep in deps:
                try:
                    print(f"📦 Installing {dep}...")
                    subprocess.run(["choco", "install", dep, "-y"], check=True)
                except subprocess.CalledProcessError:
                    print(f"⚠️ Failed to install {dep}")
        else:
            print("⚠️ Chocolatey not found. Please install ffmpeg manually.")

    def create_optimized_venv(self, config):
        """Create optimized virtual environment"""
        venv_dir = self.project_root / "nestcam_env"

        if config.get("clean_install", False) and venv_dir.exists():
            print(f"\n🧹 Removing existing virtual environment...")
            shutil.rmtree(venv_dir)

        if venv_dir.exists():
            print(f"\n✅ Using existing virtual environment: {venv_dir}")
            return

        print(f"\n🌐 Creating optimized virtual environment...")

        # Use platform-specific Python
        python_cmd = config["python_cmd"]

        try:
            # Create venv with system packages for better compatibility
            subprocess.run(
                [
                    python_cmd,
                    "-m",
                    "venv",
                    (
                        "--system-site-packages"
                        if config["system"] != "windows"
                        else "-m venv"
                    ),
                    str(venv_dir),
                ],
                check=True,
            )

            print("✅ Virtual environment created successfully")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            print("💡 Try installing Python 3.11 first")
            sys.exit(1)

    def install_python_dependencies(self, config):
        """Install Python dependencies with optimizations"""
        venv_dir = self.project_root / "nestcam_env"

        # Activate virtual environment
        if config["system"] == "windows":
            activate_script = venv_dir / "Scripts" / "activate.bat"
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            activate_script = venv_dir / "bin" / "activate"
            python_exe = venv_dir / "bin" / "python"

        print(f"\n🐍 Installing Python dependencies...")

        # Upgrade pip first
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True
        )

        # Install core dependencies
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", "-r", str(requirements)],
                check=True,
            )

        # Install platform-optimized PyTorch
        self._install_optimized_pytorch(config, python_exe)

        # Install development dependencies if requested
        if config.get("dev_mode", False):
            dev_deps = [
                "black",
                "isort",
                "mypy",
                "pytest",
                "pytest-cov",
                "jupyter",
                "notebook",
                "ipykernel",
            ]
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", *dev_deps], check=True
            )

    def _install_optimized_pytorch(self, config, python_exe):
        """Install platform-optimized PyTorch"""
        gpu_mode = config.get("gpu_mode", "auto")

        if config["system"] == "darwin":
            # macOS - Metal support
            if gpu_mode in ["auto", "metal"]:
                print("🍎 Installing PyTorch with Metal support...")
                subprocess.run(
                    [
                        str(python_exe),
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/nightly/cpu",
                    ],
                    check=True,
                )
            else:
                print("💻 Installing CPU-only PyTorch...")
                subprocess.run(
                    [
                        str(python_exe),
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                    ],
                    check=True,
                )

        elif config["system"] == "linux":
            # Linux - CUDA support
            if gpu_mode in ["auto", "cuda"]:
                print("🎯 Installing PyTorch with CUDA support...")
                subprocess.run(
                    [
                        str(python_exe),
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu121",
                    ],
                    check=True,
                )
            else:
                print("💻 Installing CPU-only PyTorch...")
                subprocess.run(
                    [
                        str(python_exe),
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                    ],
                    check=True,
                )

        else:  # Windows
            # Windows - CUDA support
            if gpu_mode in ["auto", "cuda"]:
                print("🎯 Installing PyTorch with CUDA support...")
                subprocess.run(
                    [
                        str(python_exe),
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu121",
                    ],
                    check=True,
                )
            else:
                print("💻 Installing CPU-only PyTorch...")
                subprocess.run(
                    [
                        str(python_exe),
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                    ],
                    check=True,
                )

    def configure_gpu_acceleration(self, config):
        """Configure GPU acceleration settings"""
        print("🎯 Configuring GPU acceleration...")
        venv_dir = self.project_root / "nestcam_env"

        if config["system"] == "windows":
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python"

        # Test GPU availability
        gpu_test_script = """
import torch
import platform

print(f"Platform: {platform.system()}")
print(f"PyTorch: {torch.__version__}")

gpu_info = {"backend": "cpu", "available": False}

if torch.cuda.is_available():
    gpu_info.update({
        "backend": "cuda",
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "memory": torch.cuda.get_device_properties(0).total_memory / (1024**3)
    })
elif platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    gpu_info.update({
        "backend": "metal",
        "available": True,
        "device": "Apple Silicon"
    })

import json
print(json.dumps(gpu_info))
"""

        try:
            result = subprocess.run(
                [str(python_exe), "-c", gpu_test_script],
                capture_output=True,
                text=True,
                check=True,
            )

            gpu_info = json.loads(result.stdout.strip().split("\n")[-1])

            print(f"🎯 GPU Backend: {gpu_info['backend'].upper()}")
            if gpu_info["available"]:
                print(f"✅ GPU Device: {gpu_info.get('device', 'Unknown')}")
                if "memory" in gpu_info:
                    print(f"🧠 GPU Memory: {gpu_info['memory']:.1f}GB")
            else:
                print("⚠️ GPU acceleration not available")

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            print("⚠️ Could not test GPU acceleration")

    def setup_project_config(self, config):
        """Setup project configuration"""
        print("⚙️ Setting up project configuration...")
        config_file = self.project_root / "src" / "data" / "setup_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        setup_config = {
            "platform": config["system"],
            "architecture": config["arch"],
            "gpu_backend": config.get("gpu_mode", "auto"),
            "minimal_install": config.get("minimal", False),
            "dev_mode": config.get("dev_mode", False),
            "python_version": config["recommended_python"],
            "installed_at": str(self.project_root),
            "venv_path": str(self.project_root / "nestcam_env"),
        }

        with open(config_file, "w") as f:
            json.dump(setup_config, f, indent=2)

        print("✅ Configuration saved")

    def create_launcher_scripts(self, config):
        """Create launcher scripts for easy access"""
        print("📝 Creating launcher scripts...")
        if config["system"] == "windows":
            self._create_windows_launcher(config)
        else:
            self._create_unix_launcher(config)

    def _create_unix_launcher(self, config):
        """Create Unix launcher script"""
        launcher_script = self.project_root / "launch.sh"

        script_content = f"""#!/bin/bash
# NestCam Processor v3.0 - Enhanced Launcher

PROJECT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV_DIR="$PROJECT_DIR/nestcam_env"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first: python setup.py"
    exit 1
fi

echo "🔗 Activating NestCam virtual environment..."
source "$VENV_DIR/bin/activate"

if [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    echo ""
    echo "🚀 Available commands:"
    echo "   python -m src.main --web          # Start web interface"
    echo "   python -m src.main --cli          # Start CLI mode"
    echo "   python test_installation.py       # Test installation"
    echo "   python setup.py --help           # Setup help"
    echo ""
    echo "📝 To deactivate: deactivate"
    
    # If arguments provided, run the application
    if [ $# -gt 0 ]; then
        python -m src.main "$@"
    fi
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
"""

        with open(launcher_script, "w") as f:
            f.write(script_content)

        # Make executable
        os.chmod(launcher_script, 0o755)

        print("✅ Launcher script created: launch.sh")

    def _create_windows_launcher(self, config):
        """Create Windows launcher script"""
        launcher_script = self.project_root / "launch.bat"

        script_content = f"""@echo off
REM NestCam Processor v3.0 - Enhanced Launcher

set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%nestcam_env

if not exist "%VENV_DIR%" (
    echo ❌ Virtual environment not found!
    echo Please run setup first: python setup.py
    pause
    exit /b 1
)

echo 🔗 Activating NestCam virtual environment...
call "%VENV_DIR%\\Scripts\\activate.bat"

if "%VIRTUAL_ENV%"=="%VENV_DIR%" (
    echo ✅ Virtual environment activated: %VIRTUAL_ENV%
    echo.
    echo 🚀 Available commands:
    echo    python -m src.main --web          # Start web interface
    echo    python -m src.main --cli          # Start CLI mode  
    echo    python test_installation.py       # Test installation
    echo    python setup.py --help           # Setup help
    echo.
    echo 📝 To deactivate: deactivate
    
    REM If arguments provided, run the application
    if "%~1"=="" (
        REM No arguments - show menu
        goto :show_menu
    ) else (
        REM Arguments provided - run application
        python -m src.main %*
    )
) else (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

:menu
echo.
echo Choose an option:
echo 1. Start Web Interface
echo 2. Start CLI Mode
echo 3. Test Installation
echo 4. Exit
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    python -m src.main --web
) else if "%choice%"=="2" (
    python -m src.main --cli
) else if "%choice%"=="3" (
    python test_installation.py
) else if "%choice%"=="4" (
    goto :eof
) else (
    echo Invalid choice. Please try again.
    goto :menu
)

goto :eof
"""

        with open(launcher_script, "w") as f:
            f.write(script_content)

        print("✅ Launcher script created: launch.bat")

    def run_post_install_tests(self, config):
        """Run post-installation tests"""
        print("🧪 Running post-installation tests...")
        venv_dir = self.project_root / "nestcam_env"

        if config["system"] == "windows":
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python"

        # Run test script
        test_script = self.project_root / "test_installation.py"
        if test_script.exists():
            try:
                result = subprocess.run(
                    [str(python_exe), str(test_script)], capture_output=True, text=True
                )

                if result.returncode == 0:
                    print("✅ All tests passed!")
                else:
                    print("⚠️ Some tests failed. Check output above.")
                    print(result.stdout)
                    if result.stderr:
                        print("Errors:", result.stderr)

            except subprocess.CalledProcessError as e:
                print(f"⚠️ Test execution failed: {e}")
        else:
            print("⚠️ Test script not found")

    def show_completion_message(self, config):
        """Show completion message with next steps"""
        print("🎉 NestCam Processor v3.0 Setup Complete!")
        print("=" * 50)

        if config["system"] == "windows":
            launcher_cmd = "launch.bat"
        else:
            launcher_cmd = "./launch.sh"

        print("🚀 Quick Start:")
        print(f"   {launcher_cmd} --web        # Start web interface")
        print(f"   {launcher_cmd} --cli        # Start CLI mode")
        print(f"   {launcher_cmd}              # Show menu")

        print("📋 What's New in v3.0:")
        print("   ✨ Automatic platform detection")
        print("   🎯 Optimized GPU acceleration")
        print("   🧹 Smart virtual environment management")
        print("   📊 Enhanced performance monitoring")
        print("   💾 Improved save state functionality")
        print("   🎨 Modern UI with better UX")
        if config.get("dev_mode", False):
            print("   🔧 Development tools included")
        print("📖 For help:")
        print("   python setup.py --help      # Setup options")
        print("   python test_installation.py # Test everything")
        print("🐦 Happy NestCam Processing!")
        print("=" * 50)


def main():
    """Main entry point"""
    setup = NestCamSetup()
    setup.run()


if __name__ == "__main__":
    main()
