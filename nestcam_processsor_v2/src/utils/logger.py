"""
Simple logging utilities for NestCam Processor
"""

import logging
from pathlib import Path


def setup_logging(level=logging.INFO, debug=False):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log") if debug else logging.StreamHandler(),
        ],
    )


def get_logger(name):
    """Get a logger instance"""
    return logging.getLogger(name)
