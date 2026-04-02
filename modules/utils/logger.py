"""
Logger module - Centralized logging configuration
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config


def setup_logger(name="daikoku"):
    """
    Setup and configure logger with console and file handlers

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create logs directory if it doesn't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Create formatters
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (10 MB max, keep 5 backups)
    log_filename = os.path.join(
        config.LOG_DIR,
        f"daikoku_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Create default logger instance
logger = setup_logger()


def get_logger(name=None):
    """
    Get logger instance

    Args:
        name: Optional logger name (default: uses root daikoku logger)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"daikoku.{name}")
    return logger
