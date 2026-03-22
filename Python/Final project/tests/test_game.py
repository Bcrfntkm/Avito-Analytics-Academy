"""
Unit tests for the Tic-Tac-Toe game logic.

This module contains comprehensive tests for the TicTacToeGame class,
covering all methods, edge cases, and winning conditions.
"""

import pytest

from src.game import TicTacToeGame


class TestTicTacToeGame:
    """Test suite for TicTacToeGame class."""

    def test_initialization(self):
        """Test that game initializes with empty board and correct player."""
        game = TicTacToeGame()
        board = game.get_board()

        # Check board is 3x3
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)

        # Check all cells are empty
        assert all(cell == " " for row in board for cell in row)

        # Check starting player is X
        assert game.current_player == "X"

    def test_make_valid_move(self):
        """Test making a valid move."""
        game = TicTacToeGame()

        # Make a valid move
        result = game.make_move(0, 0, "X")
        assert result is True

        # Check the move was recorded
        board = game.get_board()
        assert board[0][0] == "X"

    def test_make_move_on_occupied_cell(self):
        """Test that making a move on an occupied cell fails."""
        game = TicTacToeGame()

        # Make first move
        game.make_move(0, 0, "X")

        # Try to make move on same cell
        result = game.make_move(0, 0, "O")
        assert result is False

        # Check cell still has original value
        board = game.get_board()
        assert board[0][0] == "X"

    def test_make_move_out_of_bounds(self):
        """Test that moves outside board bounds are rejected."""
        game = TicTacToeGame()

        # Test various out of bounds positions
        assert game.make_move(-1, 0, "X") is False
        assert game.make_move(0, -1, "X") is False
        assert game.make_move(3, 0, "X") is False
        assert game.make_move(0, 3, "X") is False
        assert game.make_move(5, 5, "X") is False

    def test_make_move_invalid_player(self):
        """Test that invalid player symbols raise ValueError."""
        game = TicTacToeGame()

        with pytest.raises(ValueError, match="Invalid player symbol"):
            game.make_move(0, 0, "Z")

        with pytest.raises(ValueError, match="Invalid player symbol"):
            game.make_move(0, 0, "")

    def test_check_winner_horizontal_row_0(self):
        """Test win detection for horizontal row 0."""
        game = TicTacToeGame()

        # Create winning condition in row 0
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "X")
        game.make_move(0, 2, "X")

        assert game.check_winner() == "X"

    def test_check_winner_horizontal_row_1(self):
        """Test win detection for horizontal row 1."""
        game = TicTacToeGame()

        # Create winning condition in row 1
        game.make_move(1, 0, "O")
        game.make_move(1, 1, "O")
        game.make_move(1, 2, "O")

        assert game.check_winner() == "O"

    def test_check_winner_horizontal_row_2(self):
        """Test win detection for horizontal row 2."""
        game = TicTacToeGame()

        # Create winning condition in row 2
        game.make_move(2, 0, "X")
        game.make_move(2, 1, "X")
        game.make_move(2, 2, "X")

        assert game.check_winner() == "X"

    def test_check_winner_vertical_col_0(self):
        """Test win detection for vertical column 0."""
        game = TicTacToeGame()

        # Create winning condition in column 0
        game.make_move(0, 0, "O")
        game.make_move(1, 0, "O")
        game.make_move(2, 0, "O")

        assert game.check_winner() == "O"

    def test_check_winner_vertical_col_1(self):
        """Test win detection for vertical column 1."""
        game = TicTacToeGame()

        # Create winning condition in column 1
        game.make_move(0, 1, "X")
        game.make_move(1, 1, "X")
        game.make_move(2, 1, "X")

        assert game.check_winner() == "X"

    def test_check_winner_vertical_col_2(self):
        """Test win detection for vertical column 2."""
        game = TicTacToeGame()

        # Create winning condition in column 2
        game.make_move(0, 2, "O")
        game.make_move(1, 2, "O")
        game.make_move(2, 2, "O")

        assert game.check_winner() == "O"

    def test_check_winner_diagonal_top_left_to_bottom_right(self):
        """Test win detection for diagonal from top-left to bottom-right."""
        game = TicTacToeGame()

        # Create winning condition on main diagonal
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "X")
        game.make_move(2, 2, "X")

        assert game.check_winner() == "X"

    def test_check_winner_diagonal_top_right_to_bottom_left(self):
        """Test win detection for diagonal from top-right to bottom-left."""
        game = TicTacToeGame()

        # Create winning condition on anti-diagonal
        game.make_move(0, 2, "O")
        game.make_move(1, 1, "O")
        game.make_move(2, 0, "O")

        assert game.check_winner() == "O"

    def test_check_winner_no_winner(self):
        """Test that check_winner returns None when there's no winner."""
        game = TicTacToeGame()

        # Make some moves but no winning condition
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(1, 0, "X")

        assert game.check_winner() is None

    def test_is_draw_true(self):
        """Test draw detection when board is full with no winner."""
        game = TicTacToeGame()

        # Create a draw scenario
        # X O X
        # X O O
        # O X X
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(0, 2, "X")
        game.make_move(1, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(1, 2, "O")
        game.make_move(2, 0, "O")
        game.make_move(2, 1, "X")
        game.make_move(2, 2, "X")

        assert game.is_draw() is True

    def test_is_draw_false_with_winner(self):
        """Test that is_draw returns False when there's a winner."""
        game = TicTacToeGame()

        # Create winning condition
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "X")
        game.make_move(0, 2, "X")

        assert game.is_draw() is False

    def test_is_draw_false_with_empty_cells(self):
        """Test that is_draw returns False when board is not full."""
        game = TicTacToeGame()

        # Make some moves but leave cells empty
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")

        assert game.is_draw() is False

    def test_is_game_over_with_winner(self):
        """Test that is_game_over returns True when there's a winner."""
        game = TicTacToeGame()

        # Create winning condition
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "X")
        game.make_move(0, 2, "X")

        assert game.is_game_over() is True

    def test_is_game_over_with_draw(self):
        """Test that is_game_over returns True when it's a draw."""
        game = TicTacToeGame()

        # Create a draw scenario
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(0, 2, "X")
        game.make_move(1, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(1, 2, "O")
        game.make_move(2, 0, "O")
        game.make_move(2, 1, "X")
        game.make_move(2, 2, "X")

        assert game.is_game_over() is True

    def test_is_game_over_false(self):
        """Test that is_game_over returns False when game is ongoing."""
        game = TicTacToeGame()

        # Make some moves
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")

        assert game.is_game_over() is False

    def test_get_board_returns_copy(self):
        """Test that get_board returns a copy, not the original."""
        game = TicTacToeGame()

        # Get board and modify it
        board = game.get_board()
        board[0][0] = "X"

        # Check original board is unchanged
        original_board = game.get_board()
        assert original_board[0][0] == " "

    def test_get_board_string_empty(self):
        """Test board string representation for empty board."""
        game = TicTacToeGame()
        board_str = game.get_board_string()

        # Check format
        assert "|" in board_str
        assert "-" in board_str
        assert board_str.count("\n") == 4  # 3 rows + 2 separators

    def test_get_board_string_with_moves(self):
        """Test board string representation with moves."""
        game = TicTacToeGame()

        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(2, 2, "X")

        board_str = game.get_board_string()

        # Check that moves are in the string
        assert "X" in board_str
        assert "O" in board_str

    def test_reset(self):
        """Test that reset clears the board and resets player."""
        game = TicTacToeGame()

        # Make some moves
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(2, 2, "X")
        game.current_player = "O"

        # Reset the game
        game.reset()

        # Check board is empty
        board = game.get_board()
        assert all(cell == " " for row in board for cell in row)

        # Check player is reset to X
        assert game.current_player == "X"

    def test_get_available_moves_empty_board(self):
        """Test getting available moves on empty board."""
        game = TicTacToeGame()
        moves = game.get_available_moves()

        # Should have all 9 positions available
        assert len(moves) == 9
        assert (0, 0) in moves
        assert (2, 2) in moves

    def test_get_available_moves_partial_board(self):
        """Test getting available moves on partially filled board."""
        game = TicTacToeGame()

        # Make some moves
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(2, 2, "X")

        moves = game.get_available_moves()

        # Should have 6 positions available
        assert len(moves) == 6
        assert (0, 0) not in moves
        assert (1, 1) not in moves
        assert (2, 2) not in moves
        assert (0, 1) in moves
        assert (1, 0) in moves

    def test_get_available_moves_full_board(self):
        """Test getting available moves on full board."""
        game = TicTacToeGame()

        # Fill the board
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(0, 2, "X")
        game.make_move(1, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(1, 2, "O")
        game.make_move(2, 0, "O")
        game.make_move(2, 1, "X")
        game.make_move(2, 2, "X")

        moves = game.get_available_moves()

        # Should have no positions available
        assert len(moves) == 0

    def test_complete_game_scenario(self):
        """Test a complete game scenario from start to finish."""
        game = TicTacToeGame()

        # Play a game where X wins
        assert game.make_move(0, 0, "X") is True
        assert game.check_winner() is None
        assert game.is_game_over() is False

        assert game.make_move(1, 0, "O") is True
        assert game.check_winner() is None

        assert game.make_move(0, 1, "X") is True
        assert game.check_winner() is None

        assert game.make_move(1, 1, "O") is True
        assert game.check_winner() is None

        assert game.make_move(0, 2, "X") is True
        assert game.check_winner() == "X"
        assert game.is_game_over() is True

    def test_edge_case_winner_with_full_board(self):
        """Test that winner is detected even when board is full."""
        game = TicTacToeGame()

        # Create a scenario where board is full and X wins
        # X X X
        # O O X
        # X O O
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "X")
        game.make_move(0, 2, "X")
        game.make_move(1, 0, "O")
        game.make_move(1, 1, "O")
        game.make_move(1, 2, "X")
        game.make_move(2, 0, "X")
        game.make_move(2, 1, "O")
        game.make_move(2, 2, "O")

        assert game.check_winner() == "X"
        assert game.is_draw() is False
        assert game.is_game_over() is True
