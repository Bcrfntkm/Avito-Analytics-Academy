"""
Tests for Game Manager module.

This module contains unit tests for the GameManager class.
"""

import pytest

from src.ai_player import AIPlayer
from src.game import TicTacToeGame
from src.game_manager import GameManager


class TestGameManager:
    """Test cases for GameManager class."""

    def test_init(self):
        """Test GameManager initialization."""
        manager = GameManager()
        assert isinstance(manager.games, dict)
        assert len(manager.games) == 0

    def test_create_single_player_game_with_x(self):
        """Test creating a single-player game with X symbol."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")

        game_data = manager.get_game(user_id)
        assert game_data is not None
        assert isinstance(game_data["game"], TicTacToeGame)
        assert game_data["mode"] == "single"
        assert game_data["player_symbol"] == "X"
        assert game_data["ai_symbol"] == "O"
        assert isinstance(game_data["ai_player"], AIPlayer)
        assert "stats" in game_data

    def test_create_single_player_game_with_o(self):
        """Test creating a single-player game with O symbol."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "O")

        game_data = manager.get_game(user_id)
        assert game_data is not None
        assert game_data["player_symbol"] == "O"
        assert game_data["ai_symbol"] == "X"

    def test_create_single_player_game_invalid_symbol(self):
        """Test that creating game with invalid symbol raises ValueError."""
        manager = GameManager()
        user_id = 12345

        with pytest.raises(ValueError) as exc_info:
            manager.create_single_player_game(user_id, "Z")
        assert "Invalid player symbol" in str(exc_info.value)

    def test_get_game_existing(self):
        """Test retrieving an existing game."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        game_data = manager.get_game(user_id)

        assert game_data is not None
        assert game_data["player_symbol"] == "X"

    def test_get_game_non_existing(self):
        """Test retrieving a non-existing game returns None."""
        manager = GameManager()
        user_id = 12345

        game_data = manager.get_game(user_id)
        assert game_data is None

    def test_delete_game(self):
        """Test deleting a game."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        assert manager.get_game(user_id) is not None

        manager.delete_game(user_id)
        assert manager.get_game(user_id) is None

    def test_delete_non_existing_game(self):
        """Test deleting a non-existing game doesn't raise error."""
        manager = GameManager()
        user_id = 12345

        # Should not raise any error
        manager.delete_game(user_id)

    def test_update_stats_win(self):
        """Test updating stats with a win."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        manager.update_stats(user_id, "win")

        stats = manager.get_stats(user_id)
        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["draws"] == 0

    def test_update_stats_loss(self):
        """Test updating stats with a loss."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        manager.update_stats(user_id, "loss")

        stats = manager.get_stats(user_id)
        assert stats["wins"] == 0
        assert stats["losses"] == 1
        assert stats["draws"] == 0

    def test_update_stats_draw(self):
        """Test updating stats with a draw."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        manager.update_stats(user_id, "draw")

        stats = manager.get_stats(user_id)
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["draws"] == 1

    def test_update_stats_invalid_result(self):
        """Test that updating stats with invalid result raises ValueError."""
        manager = GameManager()
        user_id = 12345

        with pytest.raises(ValueError) as exc_info:
            manager.update_stats(user_id, "invalid")
        assert "Invalid result" in str(exc_info.value)

    def test_update_stats_multiple_games(self):
        """Test updating stats across multiple games."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        manager.update_stats(user_id, "win")
        manager.delete_game(user_id)

        manager.create_single_player_game(user_id, "O")
        manager.update_stats(user_id, "loss")
        manager.delete_game(user_id)

        manager.create_single_player_game(user_id, "X")
        manager.update_stats(user_id, "draw")

        stats = manager.get_stats(user_id)
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["draws"] == 1

    def test_get_stats_new_user(self):
        """Test getting stats for a new user returns zeros."""
        manager = GameManager()
        user_id = 12345

        stats = manager.get_stats(user_id)
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["draws"] == 0

    def test_get_stats_returns_copy(self):
        """Test that get_stats returns a copy, not reference."""
        manager = GameManager()
        user_id = 12345

        stats1 = manager.get_stats(user_id)
        stats1["wins"] = 999  # Modify the returned dict

        stats2 = manager.get_stats(user_id)
        assert stats2["wins"] == 0  # Should still be 0

    def test_multiple_users(self):
        """Test managing games for multiple users simultaneously."""
        manager = GameManager()
        user1 = 11111
        user2 = 22222
        user3 = 33333

        # Create games for multiple users
        manager.create_single_player_game(user1, "X")
        manager.create_single_player_game(user2, "O")
        manager.create_single_player_game(user3, "X")

        # Verify each user has their own game
        game1 = manager.get_game(user1)
        game2 = manager.get_game(user2)
        game3 = manager.get_game(user3)

        assert game1["player_symbol"] == "X"
        assert game2["player_symbol"] == "O"
        assert game3["player_symbol"] == "X"

        # Verify games are independent
        assert game1["game"] is not game2["game"]
        assert game2["game"] is not game3["game"]

    def test_multiple_users_stats(self):
        """Test that stats are tracked independently for multiple users."""
        manager = GameManager()
        user1 = 11111
        user2 = 22222

        # User 1 plays and wins
        manager.create_single_player_game(user1, "X")
        manager.update_stats(user1, "win")

        # User 2 plays and loses
        manager.create_single_player_game(user2, "O")
        manager.update_stats(user2, "loss")

        # Verify stats are independent
        stats1 = manager.get_stats(user1)
        stats2 = manager.get_stats(user2)

        assert stats1["wins"] == 1
        assert stats1["losses"] == 0

        assert stats2["wins"] == 0
        assert stats2["losses"] == 1

    def test_stats_persist_after_game_deletion(self):
        """Test that stats persist after game is deleted."""
        manager = GameManager()
        user_id = 12345

        # Play a game and update stats
        manager.create_single_player_game(user_id, "X")
        manager.update_stats(user_id, "win")
        manager.delete_game(user_id)

        # Stats should still be available
        stats = manager.get_stats(user_id)
        assert stats["wins"] == 1

        # Create new game and verify stats are preserved
        manager.create_single_player_game(user_id, "O")
        game_data = manager.get_game(user_id)
        assert game_data["stats"]["wins"] == 1

    def test_game_data_structure(self):
        """Test that game data has all required fields."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        game_data = manager.get_game(user_id)

        # Verify all required fields are present
        required_fields = [
            "game",
            "mode",
            "player_symbol",
            "ai_symbol",
            "ai_player",
            "stats",
        ]
        for field in required_fields:
            assert field in game_data

        # Verify stats structure
        assert "wins" in game_data["stats"]
        assert "losses" in game_data["stats"]
        assert "draws" in game_data["stats"]

    def test_concurrent_game_attempt(self):
        """Test handling of concurrent game creation for same user."""
        manager = GameManager()
        user_id = 12345

        # Create first game
        manager.create_single_player_game(user_id, "X")
        game1 = manager.get_game(user_id)

        # Create second game (should overwrite)
        manager.create_single_player_game(user_id, "O")
        game2 = manager.get_game(user_id)

        # Verify second game replaced first
        assert game2["player_symbol"] == "O"
        assert game1["game"] is not game2["game"]


class TestMultiplayerMethods:
    """Test cases for multiplayer-specific methods in GameManager."""

    def test_generate_game_code_format(self):
        """Test that generate_game_code returns correct format."""
        manager = GameManager()
        code = manager.generate_game_code()

        assert len(code) == 6
        assert code.isalnum()
        assert code.isupper()

    def test_create_multiplayer_game_basic(self):
        """Test basic multiplayer game creation."""
        manager = GameManager()
        creator_id = 12345
        game_code = "ABC123"

        result = manager.create_multiplayer_game(creator_id, "X", game_code)

        assert result == game_code
        game_data = manager.get_game_by_code(game_code)
        assert game_data is not None
        assert game_data["mode"] == "multi"

    def test_join_multiplayer_game_basic(self):
        """Test basic multiplayer game joining."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        success = manager.join_multiplayer_game(joiner_id, game_code)

        assert success is True

    def test_get_game_by_code_exists(self):
        """Test getting game by code when it exists."""
        manager = GameManager()
        creator_id = 12345
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        game_data = manager.get_game_by_code(game_code)

        assert game_data is not None
        assert game_data["game_code"] == game_code

    def test_get_game_by_code_not_exists(self):
        """Test getting game by code when it doesn't exist."""
        manager = GameManager()
        game_data = manager.get_game_by_code("NOTEXIST")

        assert game_data is None

    def test_get_multiplayer_game_exists(self):
        """Test getting multiplayer game for user."""
        manager = GameManager()
        creator_id = 12345
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        game_data = manager.get_multiplayer_game(creator_id)

        assert game_data is not None
        assert game_data["mode"] == "multi"

    def test_get_multiplayer_game_single_player(self):
        """Test that get_multiplayer_game returns None for single-player."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        game_data = manager.get_multiplayer_game(user_id)

        assert game_data is None

    def test_is_player_turn_basic(self):
        """Test checking if it's player's turn."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        # Creator (X) should go first
        assert manager.is_player_turn(creator_id, game_code) is True
        assert manager.is_player_turn(joiner_id, game_code) is False

    def test_get_opponent_id_basic(self):
        """Test getting opponent ID."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        assert manager.get_opponent_id(creator_id, game_code) == joiner_id
        assert manager.get_opponent_id(joiner_id, game_code) == creator_id

    def test_switch_turn_basic(self):
        """Test switching turns."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        # Initially creator's turn
        assert manager.is_player_turn(creator_id, game_code) is True

        # Switch turn
        manager.switch_turn(game_code)
        assert manager.is_player_turn(joiner_id, game_code) is True

    def test_get_player_symbol_basic(self):
        """Test getting player symbol."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        assert manager.get_player_symbol(creator_id, game_code) == "X"
        assert manager.get_player_symbol(joiner_id, game_code) == "O"

    def test_delete_multiplayer_game_basic(self):
        """Test deleting multiplayer game."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = "ABC123"

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        manager.delete_multiplayer_game(game_code)

        assert manager.get_game_by_code(game_code) is None
        assert manager.get_game(creator_id) is None
        assert manager.get_game(joiner_id) is None
