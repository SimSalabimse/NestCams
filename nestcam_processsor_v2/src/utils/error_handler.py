"""
Error handling utilities for NestCam Processor
"""

import logging
import traceback
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors gracefully

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """

    def wrapper(*args, **kwargs) -> Optional[Any]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    return wrapper


def log_gpu_info():
    """Log GPU information for debugging"""
    try:
        import torch

        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.info("PyTorch not available")


def validate_config(config: Any) -> bool:
    """
    Validate configuration object

    Args:
        config: Configuration object to validate

    Returns:
        True if valid, False otherwise
    """
    required_attrs = ["processing", "audio", "upload"]

    for attr in required_attrs:
        if not hasattr(config, attr):
            logger.error(f"Missing required config attribute: {attr}")
            return False

    return True
