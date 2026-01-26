"""
Configuration settings for Tic-Tac-Toe Telegram Bot.

This module centralizes all configuration constants and settings
for the bot, game logic, and logging.
"""

import logging
import os
from typing import Any, Dict


class GameSettings:
    """Game-related configuration settings."""

    # Board settings
    BOARD_SIZE: int = 3
    VALID_SYMBOLS: tuple = ("X", "O")

    # Game code settings
    GAME_CODE_LENGTH: int = 6
    GAME_CODE_CHARSET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # AI settings
    DEFAULT_AI_DIFFICULTY: str = "random"
    SUPPORTED_AI_DIFFICULTIES: tuple = ("random",)


class BotSettings:
    """Bot-related configuration settings."""

    # Timeout settings (in seconds)
    REQUEST_TIMEOUT: int = 30
    CONNECT_TIMEOUT: int = 10
    READ_TIMEOUT: int = 10
    WRITE_TIMEOUT: int = 10

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 30
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # seconds
    RETRY_BACKOFF_FACTOR: float = 2.0

    # Message settings
    MAX_MESSAGE_LENGTH: int = 4096
    BUTTON_CALLBACK_DATA_MAX_LENGTH: int = 64


class ErrorMessages:
    """User-facing error messages."""

    # Generic errors
    GENERIC_ERROR: str = "❌ An error occurred. Please try again later."
    NETWORK_ERROR: str = "❌ Network error. Please check your connection and try again."
    TIMEOUT_ERROR: str = "❌ Request timed out. Please try again."

    # Game errors
    NO_ACTIVE_GAME: str = "❌ No active game found. Use /newgame to start!"
    INVALID_MOVE: str = "❌ Invalid move! Cell is already occupied."
    NOT_YOUR_TURN: str = "⏳ It's not your turn!"
    GAME_ALREADY_EXISTS: str = (
        "⚠️ You already have an active game! "
        "Finish it first or use /newgame to start over."
    )

    # Multiplayer errors
    INVALID_GAME_CODE: str = (
        "❌ Invalid game code format! Game code must be 6 alphanumeric characters."
    )
    GAME_NOT_FOUND: str = "❌ Game not found. Please check the code and try again."
    GAME_ALREADY_STARTED: str = "❌ This game has already started."
    CANNOT_JOIN_OWN_GAME: str = "❌ You cannot join your own game."
    GAME_FULL: str = "❌ This game is already full."

    # Validation errors
    INVALID_SYMBOL: str = "❌ Invalid symbol. Must be 'X' or 'O'."
    INVALID_POSITION: str = "❌ Invalid position. Must be between 0 and 8."


class LogSettings:
    """Logging configuration settings."""

    # Log levels
    DEFAULT_LEVEL: int = logging.INFO
    FILE_LEVEL: int = logging.DEBUG
    CONSOLE_LEVEL: int = logging.INFO

    # Log format
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Log file settings
    LOG_DIR: str = "logs"
    LOG_FILE: str = "bot.log"
    ERROR_LOG_FILE: str = "errors.log"
    MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5

    # Performance logging
    ENABLE_PERFORMANCE_LOGGING: bool = True
    SLOW_OPERATION_THRESHOLD: float = 1.0  # seconds


class Environment:
    """Environment-specific settings."""

    @staticmethod
    def get_bot_token() -> str:
        """
        Get the Telegram bot token from environment variables.

        Returns:
            str: The bot token

        Raises:
            ValueError: If TELEGRAM_BOT_TOKEN is not set
        """
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN environment variable is not set. "
                "Please set it in your .env file or environment."
            )
        return token

    @staticmethod
    def is_development() -> bool:
        """
        Check if running in development mode.

        Returns:
            bool: True if in development mode
        """
        return os.getenv("ENVIRONMENT", "production").lower() == "development"

    @staticmethod
    def is_production() -> bool:
        """
        Check if running in production mode.

        Returns:
            bool: True if in production mode
        """
        return os.getenv("ENVIRONMENT", "production").lower() == "production"


class ValidationRules:
    """Validation rules for user input and game state."""

    @staticmethod
    def validate_game_code(code: str) -> bool:
        """
        Validate game code format.

        Args:
            code: The game code to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not code:
            return False
        if len(code) != GameSettings.GAME_CODE_LENGTH:
            return False
        if not code.isalnum():
            return False
        return True

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate player symbol.

        Args:
            symbol: The symbol to validate

        Returns:
            bool: True if valid, False otherwise
        """
        return symbol in GameSettings.VALID_SYMBOLS

    @staticmethod
    def validate_position(row: int, col: int) -> bool:
        """
        Validate board position.

        Args:
            row: Row index
            col: Column index

        Returns:
            bool: True if valid, False otherwise
        """
        return 0 <= row < GameSettings.BOARD_SIZE and 0 <= col < GameSettings.BOARD_SIZE


def get_config() -> Dict[str, Any]:
    """
    Get all configuration settings as a dictionary.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return {
        "game": {
            "board_size": GameSettings.BOARD_SIZE,
            "valid_symbols": GameSettings.VALID_SYMBOLS,
            "game_code_length": GameSettings.GAME_CODE_LENGTH,
            "default_ai_difficulty": GameSettings.DEFAULT_AI_DIFFICULTY,
        },
        "bot": {
            "request_timeout": BotSettings.REQUEST_TIMEOUT,
            "max_retries": BotSettings.MAX_RETRIES,
            "retry_delay": BotSettings.RETRY_DELAY,
        },
        "logging": {
            "level": LogSettings.DEFAULT_LEVEL,
            "format": LogSettings.FORMAT,
            "log_dir": LogSettings.LOG_DIR,
        },
        "environment": {
            "is_development": Environment.is_development(),
            "is_production": Environment.is_production(),
        },
    }
