"""
Tic-Tac-Toe game logic implementation.

This module contains the core game logic for a Tic-Tac-Toe game,
including board management, move validation, and win detection.
"""

from typing import List, Optional, Tuple


class TicTacToeGame:
    """
    A class representing a Tic-Tac-Toe game.

    This class manages the game state, validates moves, checks for winners,
    and provides methods to interact with the game board.

    Attributes:
        board (List[List[str]]): 3x3 grid representing the game board
        current_player (str): Current player ('X' or 'O')
    """

    def __init__(self) -> None:
        """
        Initialize a new Tic-Tac-Toe game.

        Creates an empty 3x3 board and sets the starting player to 'X'.
        """
        self.board: List[List[str]] = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player: str = "X"

    def make_move(self, row: int, col: int, player: str) -> bool:
        """
        Attempt to make a move on the board.

        Args:
            row (int): Row index (0-2)
            col (int): Column index (0-2)
            player (str): Player symbol ('X' or 'O')

        Returns:
            bool: True if move was valid and made, False otherwise

        Raises:
            ValueError: If player symbol is not 'X' or 'O'
        """
        if player not in ["X", "O"]:
            raise ValueError(f"Invalid player symbol: {player}. Must be 'X' or 'O'")

        if not (0 <= row < 3 and 0 <= col < 3):
            return False

        if self.board[row][col] != " ":
            return False

        self.board[row][col] = player
        return True

    def check_winner(self) -> Optional[str]:
        """
        Check if there is a winner.

        Checks all possible winning combinations:
        - 3 horizontal rows
        - 3 vertical columns
        - 2 diagonals

        Returns:
            Optional[str]: 'X' or 'O' if there's a winner, None otherwise
        """
        for row in range(3):
            if (
                self.board[row][0] == self.board[row][1] == self.board[row][2]
                and self.board[row][0] != " "
            ):
                return self.board[row][0]

        for col in range(3):
            if (
                self.board[0][col] == self.board[1][col] == self.board[2][col]
                and self.board[0][col] != " "
            ):
                return self.board[0][col]

        if (
            self.board[0][0] == self.board[1][1] == self.board[2][2]
            and self.board[0][0] != " "
        ):
            return self.board[0][0]

        if (
            self.board[0][2] == self.board[1][1] == self.board[2][0]
            and self.board[0][2] != " "
        ):
            return self.board[0][2]

        return None

    def is_draw(self) -> bool:
        """
        Check if the game is a draw.

        A draw occurs when the board is full and there's no winner.

        Returns:
            bool: True if the game is a draw, False otherwise
        """
        if self.check_winner() is not None:
            return False

        for row in range(3):
            for col in range(3):
                if self.board[row][col] == " ":
                    return False

        return True

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        The game is over if there's a winner or it's a draw.

        Returns:
            bool: True if game is over, False otherwise
        """
        return self.check_winner() is not None or self.is_draw()

    def get_board(self) -> List[List[str]]:
        """
        Get the current board state.

        Returns:
            List[List[str]]: A copy of the current board state
        """
        return [row[:] for row in self.board]

    def get_board_string(self) -> str:
        """
        Get a formatted string representation of the board.

        Returns:
            str: Formatted board string for display

        Example:
            ```
             X | O | X
            -----------
             O | X | O
            -----------
             X |   | O
            ```
        """
        lines = []
        for i, row in enumerate(self.board):
            formatted_row = " | ".join(cell if cell != " " else " " for cell in row)
            lines.append(f" {formatted_row}")

            if i < 2:
                lines.append("-----------")

        return "\n".join(lines)

    def reset(self) -> None:
        """
        Reset the game to initial state.

        Clears the board and resets the current player to 'X'.
        """
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"

    def get_available_moves(self) -> List[Tuple[int, int]]:
        """
        Get a list of all available moves.

        Returns:
            List[Tuple[int, int]]: List of (row, col) tuples for empty cells
        """
        available = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == " ":
                    available.append((row, col))
        return available
