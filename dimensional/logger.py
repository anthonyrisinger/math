#!/usr/bin/env python3
"""Centralized logging configuration for dimensional framework."""

import logging
import sys
from typing import Optional


def configure_logging(
    level: str = "WARNING",
    format_string: Optional[str] = None,
    stream: Optional[object] = None
) -> logging.Logger:
    """
    Configure logging for the dimensional framework.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        stream: Output stream (defaults to sys.stderr)

    Returns:
        Configured root logger for 'dimensional' namespace
    """
    # Default format for library code (no timestamps or fancy output)
    if format_string is None:
        format_string = "%(name)s:%(levelname)s: %(message)s"

    # Configure root logger for dimensional namespace
    logger = logging.getLogger("dimensional")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Add stream handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Configure default logger (WARNING level for library usage)
logger = configure_logging("WARNING")
