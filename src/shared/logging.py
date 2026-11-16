"""
Standardized Research Logging Module for EduCasheer Kashmiri AI.
"""

import logging
import sys
from typing import Optional

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str = "kashmiri_ai", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance with standardized formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
