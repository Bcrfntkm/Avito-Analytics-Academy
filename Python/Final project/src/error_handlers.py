"""
Error handling utilities for Tic-Tac-Toe Telegram Bot.

This module provides custom exceptions, error handlers, and retry logic
for robust error handling throughout the application.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, Type, Union

from telegram.error import (
    BadRequest,
    Forbidden,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)

from config.settings import BotSettings, ErrorMessages
from src.logging_config import get_logger, log_error_with_context

logger = get_logger(__name__)


# Custom Exceptions
class GameError(Exception):
    """Base exception for game-related errors."""

    pass


class InvalidMoveError(GameError):
    """Raised when an invalid move is attempted."""

    pass


class GameNotFoundError(GameError):
    """Raised when a game is not found."""

    pass


class GameStateError(GameError):
    """Raised when game state is inconsistent."""

    pass


class InvalidGameCodeError(GameError):
    """Raised when game code format is invalid."""

    pass


class GameFullError(GameError):
    """Raised when trying to join a full game."""

    pass


class NotPlayerTurnError(GameError):
    """Raised when player tries to move out of turn."""

    pass


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class InvalidSymbolError(ValidationError):
    """Raised when an invalid symbol is provided."""

    pass


class InvalidPositionError(ValidationError):
    """Raised when an invalid board position is provided."""

    pass


# Error Handler Decorators
def handle_telegram_errors(func: Callable) -> Callable:
    """
    Decorator to handle Telegram API errors gracefully.

    Catches common Telegram errors and provides user-friendly messages.

    Args:
        func: Async function to decorate

    Returns:
        Callable: Decorated function

    Example:
        ```python
        @handle_telegram_errors
        async def send_message(update, context):
            await update.message.reply_text("Hello!")
        ```
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except TimedOut as e:
            log_error_with_context(
                e,
                f"Timeout in {func.__name__}",
                function=func.__name__,
            )
            # Try to notify user if update is available
            if args and hasattr(args[0], "effective_message"):
                try:
                    await args[0].effective_message.reply_text(
                        ErrorMessages.TIMEOUT_ERROR
                    )
                except Exception:
                    pass
            raise
        except NetworkError as e:
            log_error_with_context(
                e,
                f"Network error in {func.__name__}",
                function=func.__name__,
            )
            if args and hasattr(args[0], "effective_message"):
                try:
                    await args[0].effective_message.reply_text(
                        ErrorMessages.NETWORK_ERROR
                    )
                except Exception:
                    pass
            raise
        except Forbidden as e:
            logger.warning(f"Bot was blocked by user in {func.__name__}: {e}")
            # User blocked the bot, nothing we can do
            pass
        except BadRequest as e:
            log_error_with_context(
                e,
                f"Bad request in {func.__name__}",
                function=func.__name__,
            )
            if args and hasattr(args[0], "effective_message"):
                try:
                    await args[0].effective_message.reply_text(
                        ErrorMessages.GENERIC_ERROR
                    )
                except Exception:
                    pass
        except TelegramError as e:
            log_error_with_context(
                e,
                f"Telegram error in {func.__name__}",
                function=func.__name__,
            )
            if args and hasattr(args[0], "effective_message"):
                try:
                    await args[0].effective_message.reply_text(
                        ErrorMessages.GENERIC_ERROR
                    )
                except Exception:
                    pass
            raise

    return wrapper


def handle_game_errors(func: Callable) -> Callable:
    """
    Decorator to handle game-related errors gracefully.

    Catches game errors and provides user-friendly messages.

    Args:
        func: Async function to decorate

    Returns:
        Callable: Decorated function

    Example:
        ```python
        @handle_game_errors
        async def make_move(update, context):
            # Game logic here
            pass
        ```
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except InvalidMoveError as e:
            logger.info(f"Invalid move attempted in {func.__name__}: {e}")
            if args and hasattr(args[0], "callback_query"):
                await args[0].callback_query.answer(
                    ErrorMessages.INVALID_MOVE,
                    show_alert=True,
                )
        except NotPlayerTurnError as e:
            logger.info(f"Out of turn move attempted in {func.__name__}: {e}")
            if args and hasattr(args[0], "callback_query"):
                await args[0].callback_query.answer(
                    ErrorMessages.NOT_YOUR_TURN,
                    show_alert=True,
                )
        except GameNotFoundError as e:
            logger.warning(f"Game not found in {func.__name__}: {e}")
            if args and hasattr(args[0], "effective_message"):
                await args[0].effective_message.reply_text(ErrorMessages.NO_ACTIVE_GAME)
        except InvalidGameCodeError as e:
            logger.info(f"Invalid game code in {func.__name__}: {e}")
            if args and hasattr(args[0], "effective_message"):
                await args[0].effective_message.reply_text(
                    ErrorMessages.INVALID_GAME_CODE
                )
        except GameFullError as e:
            logger.info(f"Attempted to join full game in {func.__name__}: {e}")
            if args and hasattr(args[0], "effective_message"):
                await args[0].effective_message.reply_text(ErrorMessages.GAME_FULL)
        except GameStateError as e:
            log_error_with_context(
                e,
                f"Game state error in {func.__name__}",
                function=func.__name__,
            )
            if args and hasattr(args[0], "effective_message"):
                await args[0].effective_message.reply_text(ErrorMessages.GENERIC_ERROR)
        except GameError as e:
            log_error_with_context(
                e,
                f"Game error in {func.__name__}",
                function=func.__name__,
            )
            if args and hasattr(args[0], "effective_message"):
                await args[0].effective_message.reply_text(ErrorMessages.GENERIC_ERROR)

    return wrapper


def retry_on_error(
    max_retries: int = BotSettings.MAX_RETRIES,
    delay: float = BotSettings.RETRY_DELAY,
    backoff: float = BotSettings.RETRY_BACKOFF_FACTOR,
    exceptions: tuple = (NetworkError, TimedOut),
) -> Callable:
    """
    Decorator to retry function on specific exceptions.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Callable: Decorator function

    Example:
        ```python
        @retry_on_error(max_retries=3, delay=1.0)
        async def unreliable_api_call():
            # API call that might fail
            pass
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for "
                            f"{func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for "
                            f"{func.__name__}"
                        )

            # If we get here, all retries failed
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def handle_rate_limit(func: Callable) -> Callable:
    """
    Decorator to handle Telegram rate limiting.

    Automatically waits and retries when rate limited.

    Args:
        func: Async function to decorate

    Returns:
        Callable: Decorated function

    Example:
        ```python
        @handle_rate_limit
        async def send_many_messages():
            # Send messages
            pass
        ```
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        while True:
            try:
                return await func(*args, **kwargs)
            except RetryAfter as e:
                wait_time = e.retry_after
                logger.warning(
                    f"Rate limited in {func.__name__}. Waiting {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            except TelegramError as e:
                # Check if it's a rate limit error in the message
                if "too many requests" in str(e).lower():
                    wait_time = 5  # Default wait time
                    logger.warning(
                        f"Rate limited in {func.__name__}. Waiting {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

    return wrapper


class ErrorContext:
    """
    Context manager for handling errors with automatic logging and cleanup.

    Example:
        ```python
        async with ErrorContext("processing user request", user_id=123):
            # Your code here
            await process_request()
        ```
    """

    def __init__(
        self,
        operation: str,
        cleanup_func: Optional[Callable] = None,
        **context: Any,
    ):
        """
        Initialize error context.

        Args:
            operation: Description of the operation
            cleanup_func: Optional cleanup function to call on error
            **context: Additional context information
        """
        self.operation = operation
        self.cleanup_func = cleanup_func
        self.context = context

    async def __aenter__(self) -> "ErrorContext":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit async context and handle errors."""
        if exc_type is not None:
            log_error_with_context(
                exc_val,
                self.operation,
                **self.context,
            )

            # Run cleanup if provided
            if self.cleanup_func:
                try:
                    if asyncio.iscoroutinefunction(self.cleanup_func):
                        await self.cleanup_func()
                    else:
                        self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(
                        f"Error during cleanup: {cleanup_error}",
                        exc_info=True,
                    )

            return False  # Re-raise exception

        return True


def safe_execute(
    func: Callable,
    default: Any = None,
    log_errors: bool = True,
) -> Callable:
    """
    Decorator to safely execute a function and return default on error.

    Args:
        func: Function to decorate
        default: Default value to return on error
        log_errors: Whether to log errors

    Returns:
        Callable: Decorated function

    Example:
        ```python
        @safe_execute(default={})
        def get_user_data(user_id):
            # Might raise exception
            return fetch_data(user_id)
        ```
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                log_error_with_context(
                    e,
                    f"Error in {func.__name__}",
                    function=func.__name__,
                )
            return default

    return wrapper
