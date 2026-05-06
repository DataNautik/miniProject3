"""Centralized logging configuration for Mini-project 3.

This module provides a simple logging setup that can be imported and used
throughout the project. It supports both console and file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


# ----- Module-level logger configuration -----

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Get or create a logger for a module.
    
    Args:
        name: Logger name (typically __name__).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> import logging_config
        >>> logger = logging_config.get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        # Create console handler with formatted output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    _loggers[name] = logger
    return logger


def add_file_handler(
    logger: logging.Logger,
    log_path: str | Path,
    log_level: int = logging.DEBUG,
) -> None:
    """Add a file handler to an existing logger.
    
    Args:
        logger: Logger instance to add handler to.
        log_path: Path where to write log file.
        log_level: Logging level for file output.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> add_file_handler(logger, "logs/debug.log")
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(log_level)
    
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


def set_log_level(logger: logging.Logger, level: int) -> None:
    """Change the logging level for all handlers of a logger.
    
    Args:
        logger: Logger instance.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Example:
        >>> logger = get_logger(__name__)
        >>> set_log_level(logger, logging.DEBUG)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def configure_project_logging(
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> None:
    """Configure logging for the entire project.
    
    This should be called once at program startup.
    
    Args:
        log_file: Optional path to write logs to. If None, only console output.
        level: Default logging level for all loggers.
    
    Example:
        >>> configure_project_logging(log_file="logs/run.log", level=logging.DEBUG)
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file is not None:
        add_file_handler(root_logger, log_file, log_level=level)


if __name__ == "__main__":
    # Demo usage
    configure_project_logging(level=logging.DEBUG)
    
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
