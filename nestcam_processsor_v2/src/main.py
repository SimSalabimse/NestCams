"""
Main entry point for NestCam Processor v2.0
"""

import sys
from pathlib import Path
import argparse
import logging
from .config import config


def launch_streamlit_app(web_app_path: Path, port: int = 8501, address: str = "0.0.0.0"):
    """
    Launch Streamlit application with proper configuration

    Args:
        web_app_path: Path to the Streamlit app file
        port: Port to run the server on
        address: Address to bind the server to
    """
    import os

    print("🌐 Opening web interface...")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print("❌ Press Ctrl+C to stop the server")
    # Launch Streamlit with the correct arguments
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(web_app_path),
        "--server.port",
        str(port),
        "--server.address",
        address,
    ]

    print(f"🚀 Launching: {' '.join(cmd)}")

    try:
        # Execute Streamlit (this will replace the current process)
        os.execv(sys.executable, cmd)
    except Exception as e:
        print(f"❌ Failed to launch web interface: {e}")
        print("💡 Try running manually: streamlit run src/ui/web_app.py"        sys.exit(1)


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="NestCam Processor v2.0")
    parser.add_argument("--web", action="store_true", help="Run web interface")
    parser.add_argument("--cli", action="store_true", help="Run command line interface")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=8501, help="Port for web interface")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log") if args.debug else logging.StreamHandler(),
        ],
    )

    # Load custom config if specified
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config.load_from_file(config_path)
        else:
            print(f"Warning: Config file not found: {args.config}")

    # Set debug mode
    if args.debug:
        config.debug = True

    # Run appropriate interface
    if args.web:
        print("Starting NestCam Processor v2.0 - Web Interface...")

        # Get the path to the web_app.py file
        web_app_path = Path(__file__).parent / "ui" / "web_app.py"

        if not web_app_path.exists():
            print(f"❌ Web app file not found: {web_app_path}")
            sys.exit(1)

        launch_streamlit_app(web_app_path, args.port)

    elif args.cli:
        print("NestCam Processor v2.0 - CLI mode not implemented yet")
        print("Use --web flag to run the web interface")
        print("💡 Or run: streamlit run src/ui/web_app.py"        sys.exit(1)
    else:
        # Default to web interface
        print("Starting NestCam Processor v2.0 - Web Interface...")

        # Get the path to the web_app.py file
        web_app_path = Path(__file__).parent / "ui" / "web_app.py"

        if not web_app_path.exists():
            print(f"❌ Web app file not found: {web_app_path}")
            sys.exit(1)

        launch_streamlit_app(web_app_path, args.port)


if __name__ == "__main__":
    main()
