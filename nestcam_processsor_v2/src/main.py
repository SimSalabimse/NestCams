"""
Main entry point for NestCam Processor v2.0
"""

import sys
from pathlib import Path
import argparse
import logging
from .config import config
from .ui.web_app import main as run_web_app


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="NestCam Processor v2.0")
    parser.add_argument("--web", action="store_true", help="Run web interface")
    parser.add_argument("--cli", action="store_true", help="Run command line interface")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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
        run_web_app()
    elif args.cli:
        print("NestCam Processor v2.0 - CLI mode not implemented yet")
        print("Use --web flag to run the web interface")
        sys.exit(1)
    else:
        # Default to web interface
        print("Starting NestCam Processor v2.0 - Web Interface...")
        run_web_app()


if __name__ == "__main__":
    main()
