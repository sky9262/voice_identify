# -*- coding: utf-8 -*-
"""
Logging configuration for Speaker Identification System.
"""

import logging
import sys

# Create logger
logger = logging.getLogger('voice_identify')
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers
if not logger.handlers:
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Format: timestamp - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Convenience functions that mirror print() usage
def info(msg):
    """Log info message."""
    logger.info(msg)

def debug(msg):
    """Log debug message."""
    logger.debug(msg)

def warning(msg):
    """Log warning message."""
    logger.warning(msg)

def error(msg):
    """Log error message."""
    logger.error(msg)

def set_level(level):
    """Set logging level. Use 'DEBUG', 'INFO', 'WARNING', 'ERROR'."""
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
