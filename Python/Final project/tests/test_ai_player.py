"""
Tests for AI Player module.

This module contains unit tests for the AIPlayer class.
"""

import pytest

from src.ai_player import AIPlayer
from src.game import TicTacToeGame


class TestAIPlayer:
    """Test cases for AIPlayer class."""

    def test_init_with_valid_difficulty(self):
        """Test AIPlayer initialization with valid difficulty."""
        ai = AIPlayer(difficulty="random")
        assert ai.difficulty == "random"

    def test_init_with_invalid_difficulty(self):
        """Test AIPlayer initialization with invalid difficulty raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AIPlayer(difficulty="invalid")
        assert "Unsupported difficulty" in str(exc_info.value)

    def test_init_default_difficulty(self):
        """Test AIPlayer initialization with default difficulty."""
        ai = AIPlayer()
        assert ai.difficulty == "random"

    def test_get_move_returns_valid_position(self):
        """Test that get_move returns a valid board position."""
        game = TicTacToeGame()
        ai = AIPlayer(difficulty="random")

        move = ai.get_move(game)

        # Check that move is a tuple of two integers
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert isinstance(move[0], int)
        assert isinstance(move[1], int)

        # Check that move is within board bounds
        assert 0 <= move[0] < 3
        assert 0 <= move[1] < 3

    def test_get_move_returns_empty_cell(self):
        """Test that get_move returns an empty cell."""
        game = TicTacToeGame()
        ai = AIPlayer(difficulty="random")

        # Make some moves to fill some cells
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(2, 2, "X")

        move = ai.get_move(game)
        row, col = move

        # Check that the returned position is empty
        assert game.get_board()[row][col] == " "

    def test_get_move_with_almost_full_board(self):
        """Test get_move when board has only one empty cell."""
        game = TicTacToeGame()
        ai = AIPlayer(difficulty="random")

        # Fill all cells except one
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(0, 2, "X")
        game.make_move(1, 0, "O")
        game.make_move(1, 1, "X")
        game.make_move(1, 2, "O")
        game.make_move(2, 0, "X")
        game.make_move(2, 1, "O")
        # (2, 2) is empty

        move = ai.get_move(game)

        # Should return the only available position
        assert move == (2, 2)

    def test_get_move_with_full_board_raises_error(self):
        """Test that get_move raises ValueError when board is full."""
        game = TicTacToeGame()
        ai = AIPlayer(difficulty="random")

        # Fill the entire board
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(0, 2, "X")
        game.make_move(1, 0, "O")
        game.make_move(1, 1, "X")
        game.make_move(1, 2, "O")
        game.make_move(2, 0, "X")
        game.make_move(2, 1, "O")
        game.make_move(2, 2, "X")

        with pytest.raises(ValueError) as exc_info:
            ai.get_move(game)
        assert "No available moves" in str(exc_info.value)

    def test_get_move_multiple_calls_return_valid_moves(self):
        """Test that multiple calls to get_move return valid moves."""
        game = TicTacToeGame()
        ai = AIPlayer(difficulty="random")

        # Make several moves and verify each is valid
        for _ in range(5):
            move = ai.get_move(game)
            row, col = move

            # Verify the move is valid
            assert game.get_board()[row][col] == " "

            # Make the move
            game.make_move(row, col, "X")

    def test_random_ai_picks_different_moves(self):
        """Test that random AI doesn't always pick the same move."""
        # This test has a small chance of false negative, but very unlikely
        moves = set()

        for _ in range(10):
            game = TicTacToeGame()
            ai = AIPlayer(difficulty="random")
            move = ai.get_move(game)
            moves.add(move)

        # With 9 possible positions and 10 trials, we should see variety
        # (unless extremely unlucky with random)
        assert len(moves) > 1

    def test_get_move_doesnt_pick_occupied_cells(self):
        """Test that AI never picks an occupied cell across multiple games."""
        for _ in range(20):
            game = TicTacToeGame()
            ai = AIPlayer(difficulty="random")

            # Randomly fill some cells
            import random

            num_filled = random.randint(1, 7)
            filled_positions = set()

            for _ in range(num_filled):
                available = game.get_available_moves()
                if available:
                    pos = random.choice(available)
                    game.make_move(pos[0], pos[1], "X")
                    filled_positions.add(pos)

            # Get AI move
            if game.get_available_moves():  # If board not full
                move = ai.get_move(game)
                # Verify AI didn't pick an occupied cell
                assert move not in filled_positions
                assert game.get_board()[move[0]][move[1]] == " "
