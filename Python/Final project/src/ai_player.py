"""
AI Player for Tic-Tac-Toe game.

This module contains the AI player implementation that can play against
human players with different difficulty levels.
"""

import random
from typing import Tuple

from src.game import TicTacToeGame


class AIPlayer:
    """
    AI player for Tic-Tac-Toe game.

    This class implements an AI opponent that can play Tic-Tac-Toe
    with different difficulty levels.

    Attributes:
        difficulty (str): The difficulty level of the AI ('random' for now)
    """

    def __init__(self, difficulty: str = "random") -> None:
        """
        Initialize the AI player with a specified difficulty level.

        Args:
            difficulty (str): The difficulty level. Currently supports:
                - 'random': Makes random valid moves

        Raises:
            ValueError: If an unsupported difficulty level is provided
        """
        supported_difficulties = ["random"]
        if difficulty not in supported_difficulties:
            raise ValueError(
                f"Unsupported difficulty: {difficulty}. "
                f"Supported: {', '.join(supported_difficulties)}"
            )
        self.difficulty: str = difficulty

    def get_move(self, game: TicTacToeGame) -> Tuple[int, int]:
        """
        Get the AI's chosen move for the current game state.

        Args:
            game (TicTacToeGame): The current game instance

        Returns:
            Tuple[int, int]: A tuple of (row, col) representing the chosen move

        Raises:
            ValueError: If there are no available moves (board is full)
        """
        if self.difficulty == "random":
            return self._get_random_move(game)

        raise ValueError(f"Unsupported difficulty: {self.difficulty}")

    def _get_random_move(self, game: TicTacToeGame) -> Tuple[int, int]:
        """
        Get a random valid move from available positions.

        Args:
            game (TicTacToeGame): The current game instance

        Returns:
            Tuple[int, int]: A random available position as (row, col)

        Raises:
            ValueError: If there are no available moves
        """
        available_moves = game.get_available_moves()

        if not available_moves:
            raise ValueError("No available moves on the board")

        return random.choice(available_moves)
