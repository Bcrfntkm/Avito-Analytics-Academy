"""
Tests for error handling in Tic-Tac-Toe bot.

This module tests various error scenarios including invalid inputs,
game state errors, network errors, and edge cases.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, NetworkError, TimedOut

from config.settings import ErrorMessages, ValidationRules
from src.error_handlers import (
    GameError,
    GameFullError,
    GameNotFoundError,
    GameStateError,
    InvalidGameCodeError,
    InvalidMoveError,
    InvalidSymbolError,
    NotPlayerTurnError,
)
from src.game import TicTacToeGame
from src.game_manager import GameManager


class TestValidationRules:
    """Test validation rules for user input."""

    def test_validate_game_code_valid(self):
        """Test validation of valid game codes."""
        assert ValidationRules.validate_game_code("ABC123")
        assert ValidationRules.validate_game_code("XYZ789")
        assert ValidationRules.validate_game_code("000000")

    def test_validate_game_code_invalid_length(self):
        """Test validation rejects codes with wrong length."""
        assert not ValidationRules.validate_game_code("ABC")
        assert not ValidationRules.validate_game_code("ABCDEFG")
        assert not ValidationRules.validate_game_code("")

    def test_validate_game_code_invalid_characters(self):
        """Test validation rejects codes with invalid characters."""
        assert not ValidationRules.validate_game_code("ABC-12")
        assert not ValidationRules.validate_game_code("ABC 12")
        assert not ValidationRules.validate_game_code("ABC@12")

    def test_validate_game_code_none(self):
        """Test validation rejects None."""
        assert not ValidationRules.validate_game_code(None)

    def test_validate_symbol_valid(self):
        """Test validation of valid symbols."""
        assert ValidationRules.validate_symbol("X")
        assert ValidationRules.validate_symbol("O")

    def test_validate_symbol_invalid(self):
        """Test validation rejects invalid symbols."""
        assert not ValidationRules.validate_symbol("A")
        assert not ValidationRules.validate_symbol("x")
        assert not ValidationRules.validate_symbol("")
        assert not ValidationRules.validate_symbol("XO")

    def test_validate_position_valid(self):
        """Test validation of valid board positions."""
        assert ValidationRules.validate_position(0, 0)
        assert ValidationRules.validate_position(1, 1)
        assert ValidationRules.validate_position(2, 2)

    def test_validate_position_invalid(self):
        """Test validation rejects invalid positions."""
        assert not ValidationRules.validate_position(-1, 0)
        assert not ValidationRules.validate_position(0, -1)
        assert not ValidationRules.validate_position(3, 0)
        assert not ValidationRules.validate_position(0, 3)
        assert not ValidationRules.validate_position(5, 5)


class TestGameErrors:
    """Test game-related error scenarios."""

    def test_invalid_move_occupied_cell(self):
        """Test error when trying to move to occupied cell."""
        game = TicTacToeGame()
        game.make_move(0, 0, "X")

        # Try to move to same cell
        result = game.make_move(0, 0, "O")
        assert result is False

    def test_invalid_move_out_of_bounds(self):
        """Test error when trying to move out of bounds."""
        game = TicTacToeGame()

        assert not game.make_move(-1, 0, "X")
        assert not game.make_move(0, -1, "X")
        assert not game.make_move(3, 0, "X")
        assert not game.make_move(0, 3, "X")

    def test_invalid_symbol(self):
        """Test error when using invalid symbol."""
        game = TicTacToeGame()

        with pytest.raises(ValueError, match="Invalid player symbol"):
            game.make_move(0, 0, "A")

    def test_game_not_found(self):
        """Test error when game doesn't exist."""
        manager = GameManager()

        game_data = manager.get_game(99999)
        assert game_data is None

    def test_invalid_game_code_format(self):
        """Test error with invalid game code format."""
        manager = GameManager()

        # Try to join with invalid code
        result = manager.join_multiplayer_game(123, "INVALID")
        assert result is False


class TestMultiplayerErrors:
    """Test multiplayer-specific error scenarios."""

    def test_join_nonexistent_game(self):
        """Test joining a game that doesn't exist."""
        manager = GameManager()

        result = manager.join_multiplayer_game(123, "NOEXST")
        assert result is False

    def test_join_own_game(self):
        """Test that creator cannot join their own game."""
        manager = GameManager()
        creator_id = 123

        # Create game
        code = manager.generate_game_code()
        manager.create_multiplayer_game(creator_id, "X", code)

        # Try to join own game
        result = manager.join_multiplayer_game(creator_id, code)
        assert result is False

    def test_join_already_started_game(self):
        """Test joining a game that already started."""
        manager = GameManager()
        creator_id = 123
        joiner_id = 456

        # Create and join game
        code = manager.generate_game_code()
        manager.create_multiplayer_game(creator_id, "X", code)
        manager.join_multiplayer_game(joiner_id, code)

        # Try to join again with different user
        result = manager.join_multiplayer_game(789, code)
        assert result is False

    def test_move_when_not_turn(self):
        """Test making a move when it's not player's turn."""
        manager = GameManager()
        creator_id = 123
        joiner_id = 456

        # Create and join game
        code = manager.generate_game_code()
        manager.create_multiplayer_game(creator_id, "X", code)
        manager.join_multiplayer_game(joiner_id, code)

        # X goes first, so joiner (O) shouldn't be able to move
        is_turn = manager.is_player_turn(joiner_id, code)
        assert is_turn is False


class TestConcurrentGameAttempts:
    """Test concurrent game creation and management."""

    def test_multiple_active_games_same_user(self):
        """Test that user can only have one active game."""
        manager = GameManager()
        user_id = 123

        # Create first game
        manager.create_single_player_game(user_id, "X")
        game1 = manager.get_game(user_id)
        assert game1 is not None

        # Delete and create second game
        manager.delete_game(user_id)
        manager.create_single_player_game(user_id, "O")
        game2 = manager.get_game(user_id)
        assert game2 is not None
        assert game2["player_symbol"] == "O"

    def test_unique_game_codes(self):
        """Test that generated game codes are unique."""
        manager = GameManager()

        codes = set()
        for _ in range(100):
            code = manager.generate_game_code()
            assert code not in codes
            codes.add(code)
            assert len(code) == 6
            assert code.isalnum()


class TestGameStateConsistency:
    """Test game state consistency and transitions."""

    def test_game_state_after_win(self):
        """Test game state is consistent after win."""
        game = TicTacToeGame()

        # Create winning condition for X
        game.make_move(0, 0, "X")
        game.make_move(1, 0, "O")
        game.make_move(0, 1, "X")
        game.make_move(1, 1, "O")
        game.make_move(0, 2, "X")

        assert game.check_winner() == "X"
        assert game.is_game_over()
        assert not game.is_draw()

    def test_game_state_after_draw(self):
        """Test game state is consistent after draw."""
        game = TicTacToeGame()

        # Create draw condition
        moves = [
            (0, 0, "X"),
            (0, 1, "O"),
            (0, 2, "X"),
            (1, 0, "O"),
            (1, 1, "X"),
            (1, 2, "O"),
            (2, 0, "O"),
            (2, 1, "X"),
            (2, 2, "O"),
        ]

        for row, col, symbol in moves:
            game.make_move(row, col, symbol)

        assert game.check_winner() is None
        assert game.is_draw()
        assert game.is_game_over()

    def test_stats_update_consistency(self):
        """Test that stats are updated consistently."""
        manager = GameManager()
        user_id = 123

        # Initial stats
        stats = manager.get_stats(user_id)
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["draws"] == 0

        # Update stats
        manager.update_stats(user_id, "win")
        manager.update_stats(user_id, "loss")
        manager.update_stats(user_id, "draw")

        stats = manager.get_stats(user_id)
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["draws"] == 1

    def test_invalid_stats_result(self):
        """Test error with invalid stats result."""
        manager = GameManager()

        with pytest.raises(ValueError, match="Invalid result"):
            manager.update_stats(123, "invalid")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_board_available_moves(self):
        """Test available moves on empty board."""
        game = TicTacToeGame()
        moves = game.get_available_moves()

        assert len(moves) == 9
        assert (0, 0) in moves
        assert (2, 2) in moves

    def test_full_board_available_moves(self):
        """Test available moves on full board."""
        game = TicTacToeGame()

        # Fill board
        moves = [
            (0, 0, "X"),
            (0, 1, "O"),
            (0, 2, "X"),
            (1, 0, "O"),
            (1, 1, "X"),
            (1, 2, "O"),
            (2, 0, "O"),
            (2, 1, "X"),
            (2, 2, "O"),
        ]

        for row, col, symbol in moves:
            game.make_move(row, col, symbol)

        available = game.get_available_moves()
        assert len(available) == 0

    def test_game_reset(self):
        """Test game reset functionality."""
        game = TicTacToeGame()

        # Make some moves
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")

        # Reset
        game.reset()

        # Check board is empty
        board = game.get_board()
        for row in board:
            for cell in row:
                assert cell == " "

        assert game.current_player == "X"

    def test_multiplayer_game_cleanup(self):
        """Test proper cleanup of multiplayer games."""
        manager = GameManager()
        creator_id = 123
        joiner_id = 456

        # Create and join game
        code = manager.generate_game_code()
        manager.create_multiplayer_game(creator_id, "X", code)
        manager.join_multiplayer_game(joiner_id, code)

        # Delete game
        manager.delete_multiplayer_game(code)

        # Verify cleanup
        assert manager.get_game_by_code(code) is None
        assert manager.get_game(creator_id) is None
        assert manager.get_game(joiner_id) is None


class TestErrorMessages:
    """Test error message constants."""

    def test_error_messages_exist(self):
        """Test that all error messages are defined."""
        assert ErrorMessages.GENERIC_ERROR
        assert ErrorMessages.NETWORK_ERROR
        assert ErrorMessages.TIMEOUT_ERROR
        assert ErrorMessages.NO_ACTIVE_GAME
        assert ErrorMessages.INVALID_MOVE
        assert ErrorMessages.NOT_YOUR_TURN
        assert ErrorMessages.INVALID_GAME_CODE
        assert ErrorMessages.GAME_NOT_FOUND

    def test_error_messages_are_strings(self):
        """Test that error messages are strings."""
        assert isinstance(ErrorMessages.GENERIC_ERROR, str)
        assert isinstance(ErrorMessages.NETWORK_ERROR, str)
        assert isinstance(ErrorMessages.TIMEOUT_ERROR, str)


@pytest.mark.asyncio
class TestTelegramErrorHandling:
    """Test Telegram API error handling."""

    async def test_network_error_handling(self):
        """Test handling of network errors."""
        from src.error_handlers import handle_telegram_errors

        @handle_telegram_errors
        async def failing_function():
            raise NetworkError("Connection failed")

        with pytest.raises(NetworkError):
            await failing_function()

    async def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        from src.error_handlers import handle_telegram_errors

        @handle_telegram_errors
        async def timeout_function():
            raise TimedOut("Request timed out")

        with pytest.raises(TimedOut):
            await timeout_function()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
