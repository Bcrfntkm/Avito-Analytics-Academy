"""
Logging configuration for Tic-Tac-Toe Telegram Bot.

This module provides structured logging with rotation, separate error logs,
and performance tracking capabilities.
"""

import logging
import os
import sys
import time
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import LogSettings


def setup_logging() -> logging.Logger:
    """
    Set up logging configuration with file rotation and console output.

    Creates log directory if it doesn't exist, sets up rotating file handlers
    for general logs and error logs, and configures console output.

    Returns:
        logging.Logger: Configured root logger
    """
    log_dir = Path(LogSettings.LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        LogSettings.FORMAT,
        datefmt=LogSettings.DATE_FORMAT,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(LogSettings.DEFAULT_LEVEL)

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LogSettings.CONSOLE_LEVEL)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file_path = log_dir / LogSettings.LOG_FILE
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=LogSettings.MAX_BYTES,
        backupCount=LogSettings.BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(LogSettings.FILE_LEVEL)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    error_log_path = log_dir / LogSettings.ERROR_LOG_FILE
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=LogSettings.MAX_BYTES,
        backupCount=LogSettings.BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    root_logger.info("Logging system initialized")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (typically __name__)

    Returns:
        logging.Logger: Logger instance for the module
    """
    return logging.getLogger(name)


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function execution time.

    Logs a warning if execution time exceeds the slow operation threshold.

    Args:
        func: Function to decorate

    Returns:
        Callable: Decorated function

    Example:
        ```python
        @log_performance
        def slow_operation():
            time.sleep(2)
        ```
    """
    if not LogSettings.ENABLE_PERFORMANCE_LOGGING:
        return func

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_time = time.time() - start_time
            if elapsed_time > LogSettings.SLOW_OPERATION_THRESHOLD:
                logger.warning(
                    f"Slow operation: {func.__name__} took {elapsed_time:.2f}s"
                )
            else:
                logger.debug(f"Performance: {func.__name__} took {elapsed_time:.4f}s")

    return wrapper


def log_async_performance(func: Callable) -> Callable:
    """
    Decorator to log async function execution time.

    Logs a warning if execution time exceeds the slow operation threshold.

    Args:
        func: Async function to decorate

    Returns:
        Callable: Decorated async function

    Example:
        ```python
        @log_async_performance
        async def slow_async_operation():
            await asyncio.sleep(2)
        ```
    """
    if not LogSettings.ENABLE_PERFORMANCE_LOGGING:
        return func

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed_time = time.time() - start_time
            if elapsed_time > LogSettings.SLOW_OPERATION_THRESHOLD:
                logger.warning(
                    f"Slow async operation: {func.__name__} took {elapsed_time:.2f}s"
                )
            else:
                logger.debug(f"Performance: {func.__name__} took {elapsed_time:.4f}s")

    return wrapper


class ContextLogger:
    """
    Context manager for logging operations with automatic error handling.

    Example:
        ```python
        with ContextLogger("Processing user request", user_id=123):
            # Your code here
            process_request()
        ```
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[logging.Logger] = None,
        **context: Any,
    ):
        """
        Initialize context logger.

        Args:
            operation: Description of the operation
            logger: Logger instance (uses root logger if None)
            **context: Additional context to log
        """
        self.operation = operation
        self.logger = logger or logging.getLogger()
        self.context = context
        self.start_time: Optional[float] = None

    def __enter__(self) -> "ContextLogger":
        """Enter context and log start of operation."""
        self.start_time = time.time()
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        self.logger.info(f"Starting: {self.operation} ({context_str})")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context and log completion or error."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation} after {elapsed:.2f}s - "
                f"{exc_type.__name__}: {exc_val}",
                exc_info=True,
            )
            return False

        self.logger.info(f"Completed: {self.operation} in {elapsed:.2f}s")
        return True


def log_user_action(
    action: str,
    user_id: int,
    logger: Optional[logging.Logger] = None,
    **details: Any,
) -> None:
    """
    Log a user action with structured information.

    Args:
        action: Description of the action
        user_id: Telegram user ID
        logger: Logger instance (uses root logger if None)
        **details: Additional details to log
    """
    logger = logger or logging.getLogger()
    details_str = ", ".join(f"{k}={v}" for k, v in details.items())
    logger.info(f"User action: {action} | user_id={user_id} | {details_str}")


def log_game_event(
    event: str,
    game_id: str,
    logger: Optional[logging.Logger] = None,
    **details: Any,
) -> None:
    """
    Log a game event with structured information.

    Args:
        event: Description of the event
        game_id: Game identifier (user_id or game_code)
        logger: Logger instance (uses root logger if None)
        **details: Additional details to log
    """
    logger = logger or logging.getLogger()
    details_str = ", ".join(f"{k}={v}" for k, v in details.items())
    logger.info(f"Game event: {event} | game_id={game_id} | {details_str}")


def log_error_with_context(
    error: Exception,
    context: str,
    logger: Optional[logging.Logger] = None,
    **details: Any,
) -> None:
    """
    Log an error with additional context information.

    Args:
        error: The exception that occurred
        context: Description of what was being done when error occurred
        logger: Logger instance (uses root logger if None)
        **details: Additional context details
    """
    logger = logger or logging.getLogger()
    details_str = ", ".join(f"{k}={v}" for k, v in details.items())
    logger.error(
        f"Error in {context}: {type(error).__name__}: {error} | {details_str}",
        exc_info=True,
    )
