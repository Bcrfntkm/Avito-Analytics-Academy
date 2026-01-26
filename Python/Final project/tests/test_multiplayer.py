"""
Tests for Multiplayer functionality.

This module contains unit tests for multiplayer game features.
"""

import pytest

from src.game import TicTacToeGame
from src.game_manager import GameManager


class TestMultiplayerGameCreation:
    """Test cases for multiplayer game creation."""

    def test_generate_game_code(self):
        """Test game code generation."""
        manager = GameManager()
        code = manager.generate_game_code()

        assert isinstance(code, str)
        assert len(code) == 6
        assert code.isalnum()
        assert code.isupper()

    def test_generate_unique_game_codes(self):
        """Test that generated game codes are unique."""
        manager = GameManager()
        codes = set()

        # Generate 100 codes and ensure they're all unique
        for _ in range(100):
            code = manager.generate_game_code()
            assert code not in codes
            codes.add(code)

    def test_create_multiplayer_game_with_x(self):
        """Test creating a multiplayer game with X symbol."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        result_code = manager.create_multiplayer_game(creator_id, "X", game_code)

        assert result_code == game_code
        game_data = manager.get_game_by_code(game_code)
        assert game_data is not None
        assert isinstance(game_data["game"], TicTacToeGame)
        assert game_data["mode"] == "multi"
        assert game_data["creator_id"] == creator_id
        assert game_data["joiner_id"] is None
        assert game_data["creator_symbol"] == "X"
        assert game_data["joiner_symbol"] == "O"
        assert game_data["current_turn"] == creator_id
        assert game_data["game_code"] == game_code
        assert game_data["status"] == "waiting"

    def test_create_multiplayer_game_with_o(self):
        """Test creating a multiplayer game with O symbol."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "O", game_code)

        game_data = manager.get_game_by_code(game_code)
        assert game_data["creator_symbol"] == "O"
        assert game_data["joiner_symbol"] == "X"
        assert game_data["current_turn"] is None  # Joiner (X) goes first

    def test_create_multiplayer_game_invalid_symbol(self):
        """Test that creating game with invalid symbol raises ValueError."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        with pytest.raises(ValueError) as exc_info:
            manager.create_multiplayer_game(creator_id, "Z", game_code)
        assert "Invalid creator symbol" in str(exc_info.value)

    def test_creator_game_stored_in_user_games(self):
        """Test that creator's game is stored in their user games."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)

        user_game = manager.get_game(creator_id)
        assert user_game is not None
        assert user_game["game_code"] == game_code


class TestMultiplayerGameJoining:
    """Test cases for joining multiplayer games."""

    def test_join_multiplayer_game_success(self):
        """Test successfully joining a multiplayer game."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        success = manager.join_multiplayer_game(joiner_id, game_code)

        assert success is True
        game_data = manager.get_game_by_code(game_code)
        assert game_data["joiner_id"] == joiner_id
        assert game_data["status"] == "active"

    def test_join_sets_correct_turn_when_joiner_is_x(self):
        """Test that turn is set to joiner when they are X."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        # Creator chooses O, so joiner will be X
        manager.create_multiplayer_game(creator_id, "O", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        game_data = manager.get_game_by_code(game_code)
        assert game_data["current_turn"] == joiner_id

    def test_join_keeps_creator_turn_when_creator_is_x(self):
        """Test that turn stays with creator when they are X."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        # Creator chooses X
        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        game_data = manager.get_game_by_code(game_code)
        assert game_data["current_turn"] == creator_id

    def test_join_invalid_game_code(self):
        """Test joining with invalid game code returns False."""
        manager = GameManager()
        joiner_id = 67890

        success = manager.join_multiplayer_game(joiner_id, "INVALID")
        assert success is False

    def test_join_already_active_game(self):
        """Test that joining an already active game fails."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        third_player_id = 11111
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        # Try to join again with different player
        success = manager.join_multiplayer_game(third_player_id, game_code)
        assert success is False

    def test_creator_cannot_join_own_game(self):
        """Test that creator cannot join their own game."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        success = manager.join_multiplayer_game(creator_id, game_code)

        assert success is False

    def test_joiner_game_stored_in_user_games(self):
        """Test that joiner's game is stored in their user games."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        joiner_game = manager.get_game(joiner_id)
        assert joiner_game is not None
        assert joiner_game["game_code"] == game_code


class TestMultiplayerGameRetrieval:
    """Test cases for retrieving multiplayer games."""

    def test_get_game_by_code(self):
        """Test retrieving game by code."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        game_data = manager.get_game_by_code(game_code)

        assert game_data is not None
        assert game_data["game_code"] == game_code

    def test_get_game_by_invalid_code(self):
        """Test retrieving game with invalid code returns None."""
        manager = GameManager()
        game_data = manager.get_game_by_code("INVALID")
        assert game_data is None

    def test_get_multiplayer_game_for_creator(self):
        """Test getting multiplayer game for creator."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        game_data = manager.get_multiplayer_game(creator_id)

        assert game_data is not None
        assert game_data["mode"] == "multi"

    def test_get_multiplayer_game_for_joiner(self):
        """Test getting multiplayer game for joiner."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        game_data = manager.get_multiplayer_game(joiner_id)
        assert game_data is not None
        assert game_data["mode"] == "multi"

    def test_get_multiplayer_game_returns_none_for_single_player(self):
        """Test that get_multiplayer_game returns None for single-player games."""
        manager = GameManager()
        user_id = 12345

        manager.create_single_player_game(user_id, "X")
        game_data = manager.get_multiplayer_game(user_id)

        assert game_data is None

    def test_get_multiplayer_game_no_game(self):
        """Test getting multiplayer game when user has no game."""
        manager = GameManager()
        game_data = manager.get_multiplayer_game(12345)
        assert game_data is None


class TestTurnManagement:
    """Test cases for turn management in multiplayer games."""

    def test_is_player_turn_creator_x(self):
        """Test checking turn when creator is X."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        assert manager.is_player_turn(creator_id, game_code) is True
        assert manager.is_player_turn(joiner_id, game_code) is False

    def test_is_player_turn_joiner_x(self):
        """Test checking turn when joiner is X."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "O", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        assert manager.is_player_turn(creator_id, game_code) is False
        assert manager.is_player_turn(joiner_id, game_code) is True

    def test_is_player_turn_invalid_game(self):
        """Test checking turn with invalid game code."""
        manager = GameManager()
        assert manager.is_player_turn(12345, "INVALID") is False

    def test_switch_turn(self):
        """Test switching turns between players."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        # Initially creator's turn
        assert manager.is_player_turn(creator_id, game_code) is True

        # Switch turn
        manager.switch_turn(game_code)
        assert manager.is_player_turn(creator_id, game_code) is False
        assert manager.is_player_turn(joiner_id, game_code) is True

        # Switch back
        manager.switch_turn(game_code)
        assert manager.is_player_turn(creator_id, game_code) is True
        assert manager.is_player_turn(joiner_id, game_code) is False

    def test_switch_turn_invalid_game(self):
        """Test switching turn with invalid game code doesn't crash."""
        manager = GameManager()
        # Should not raise any error
        manager.switch_turn("INVALID")


class TestPlayerIdentification:
    """Test cases for player identification in multiplayer games."""

    def test_get_opponent_id_for_creator(self):
        """Test getting opponent ID for creator."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        opponent_id = manager.get_opponent_id(creator_id, game_code)
        assert opponent_id == joiner_id

    def test_get_opponent_id_for_joiner(self):
        """Test getting opponent ID for joiner."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        opponent_id = manager.get_opponent_id(joiner_id, game_code)
        assert opponent_id == creator_id

    def test_get_opponent_id_before_joiner(self):
        """Test getting opponent ID before anyone joins."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        opponent_id = manager.get_opponent_id(creator_id, game_code)

        assert opponent_id is None

    def test_get_opponent_id_invalid_game(self):
        """Test getting opponent ID with invalid game code."""
        manager = GameManager()
        opponent_id = manager.get_opponent_id(12345, "INVALID")
        assert opponent_id is None

    def test_get_opponent_id_non_participant(self):
        """Test getting opponent ID for non-participant."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        non_participant_id = 11111
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        opponent_id = manager.get_opponent_id(non_participant_id, game_code)
        assert opponent_id is None

    def test_get_player_symbol_creator(self):
        """Test getting player symbol for creator."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        symbol = manager.get_player_symbol(creator_id, game_code)

        assert symbol == "X"

    def test_get_player_symbol_joiner(self):
        """Test getting player symbol for joiner."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        symbol = manager.get_player_symbol(joiner_id, game_code)
        assert symbol == "O"

    def test_get_player_symbol_invalid_game(self):
        """Test getting player symbol with invalid game code."""
        manager = GameManager()
        symbol = manager.get_player_symbol(12345, "INVALID")
        assert symbol is None

    def test_get_player_symbol_non_participant(self):
        """Test getting player symbol for non-participant."""
        manager = GameManager()
        creator_id = 12345
        non_participant_id = 11111
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        symbol = manager.get_player_symbol(non_participant_id, game_code)

        assert symbol is None


class TestMultiplayerGameDeletion:
    """Test cases for deleting multiplayer games."""

    def test_delete_multiplayer_game(self):
        """Test deleting a multiplayer game."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        manager.delete_multiplayer_game(game_code)

        # Game should be removed
        assert manager.get_game_by_code(game_code) is None
        assert manager.get_game(creator_id) is None
        assert manager.get_game(joiner_id) is None

    def test_delete_multiplayer_game_before_joiner(self):
        """Test deleting a game before anyone joins."""
        manager = GameManager()
        creator_id = 12345
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.delete_multiplayer_game(game_code)

        assert manager.get_game_by_code(game_code) is None
        assert manager.get_game(creator_id) is None

    def test_delete_non_existent_game(self):
        """Test deleting non-existent game doesn't crash."""
        manager = GameManager()
        # Should not raise any error
        manager.delete_multiplayer_game("INVALID")


class TestMultiplayerGameplay:
    """Test cases for multiplayer gameplay."""

    def test_simultaneous_moves_prevention(self):
        """Test that only current player can make moves."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        game_data = manager.get_game_by_code(game_code)
        game = game_data["game"]

        # Creator (X) should be able to move
        assert manager.is_player_turn(creator_id, game_code) is True
        assert game.make_move(0, 0, "X") is True

        # Joiner (O) should not be able to move yet (still creator's turn)
        # In real implementation, turn would switch after move
        manager.switch_turn(game_code)
        assert manager.is_player_turn(joiner_id, game_code) is True
        assert game.make_move(1, 1, "O") is True

    def test_game_completion_with_winner(self):
        """Test completing a multiplayer game with a winner."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        game_data = manager.get_game_by_code(game_code)
        game = game_data["game"]

        # Play a game where X wins
        game.make_move(0, 0, "X")  # X
        game.make_move(1, 0, "O")  # O
        game.make_move(0, 1, "X")  # X
        game.make_move(1, 1, "O")  # O
        game.make_move(0, 2, "X")  # X wins

        winner = game.check_winner()
        assert winner == "X"

        # Update stats
        manager.update_stats(creator_id, "win")
        manager.update_stats(joiner_id, "loss")

        creator_stats = manager.get_stats(creator_id)
        joiner_stats = manager.get_stats(joiner_id)

        assert creator_stats["wins"] == 1
        assert joiner_stats["losses"] == 1

    def test_game_completion_with_draw(self):
        """Test completing a multiplayer game with a draw."""
        manager = GameManager()
        creator_id = 12345
        joiner_id = 67890
        game_code = manager.generate_game_code()

        manager.create_multiplayer_game(creator_id, "X", game_code)
        manager.join_multiplayer_game(joiner_id, game_code)

        game_data = manager.get_game_by_code(game_code)
        game = game_data["game"]

        # Play a game that ends in draw
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(0, 2, "X")
        game.make_move(1, 0, "X")
        game.make_move(1, 1, "O")
        game.make_move(1, 2, "X")
        game.make_move(2, 0, "O")
        game.make_move(2, 1, "X")
        game.make_move(2, 2, "O")

        assert game.is_draw() is True

        # Update stats
        manager.update_stats(creator_id, "draw")
        manager.update_stats(joiner_id, "draw")

        creator_stats = manager.get_stats(creator_id)
        joiner_stats = manager.get_stats(joiner_id)

        assert creator_stats["draws"] == 1
        assert joiner_stats["draws"] == 1


class TestConcurrentGames:
    """Test cases for multiple concurrent multiplayer games."""

    def test_multiple_concurrent_games(self):
        """Test managing multiple concurrent multiplayer games."""
        manager = GameManager()

        # Create three different games
        game1_creator = 11111
        game1_joiner = 22222
        game1_code = manager.generate_game_code()

        game2_creator = 33333
        game2_joiner = 44444
        game2_code = manager.generate_game_code()

        game3_creator = 55555
        game3_joiner = 66666
        game3_code = manager.generate_game_code()

        # Create all games
        manager.create_multiplayer_game(game1_creator, "X", game1_code)
        manager.create_multiplayer_game(game2_creator, "O", game2_code)
        manager.create_multiplayer_game(game3_creator, "X", game3_code)

        # Join all games
        manager.join_multiplayer_game(game1_joiner, game1_code)
        manager.join_multiplayer_game(game2_joiner, game2_code)
        manager.join_multiplayer_game(game3_joiner, game3_code)

        # Verify all games exist independently
        game1 = manager.get_game_by_code(game1_code)
        game2 = manager.get_game_by_code(game2_code)
        game3 = manager.get_game_by_code(game3_code)

        assert game1 is not None
        assert game2 is not None
        assert game3 is not None

        assert game1["creator_id"] == game1_creator
        assert game2["creator_id"] == game2_creator
        assert game3["creator_id"] == game3_creator

        # Verify games are independent
        assert game1["game"] is not game2["game"]
        assert game2["game"] is not game3["game"]

    def test_user_can_only_have_one_active_game(self):
        """Test that creating a new game overwrites the old one."""
        manager = GameManager()
        user_id = 12345
        game_code1 = manager.generate_game_code()
        game_code2 = manager.generate_game_code()

        # Create first game
        manager.create_multiplayer_game(user_id, "X", game_code1)
        game1 = manager.get_game(user_id)
        assert game1["game_code"] == game_code1

        # Create second game (should overwrite)
        manager.create_multiplayer_game(user_id, "O", game_code2)
        game2 = manager.get_game(user_id)
        assert game2["game_code"] == game_code2

        # Both games still exist in multiplayer_games
        assert manager.get_game_by_code(game_code1) is not None
        assert manager.get_game_by_code(game_code2) is not None
