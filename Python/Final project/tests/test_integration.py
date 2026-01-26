"""
Integration tests for Tic-Tac-Toe Telegram Bot.

This module contains end-to-end integration tests that verify
complete game flows, stats persistence, and error recovery scenarios.
"""

import pytest

from config.settings import ErrorMessages, GameSettings
from src.ai_player import AIPlayer
from src.game import TicTacToeGame
from src.game_manager import GameManager


class TestSinglePlayerIntegration:
    """Integration tests for single-player game flow."""

    def test_complete_single_player_game_x_wins(self):
        """Test complete single-player game where player X wins."""
        manager = GameManager()
        user_id = 12345

        # Create game with player as X
        game_data = manager.create_single_player_game(user_id, "X")
        assert game_data is not None
        assert game_data["player_symbol"] == "X"
        assert game_data["ai_symbol"] == "O"

        game = game_data["game"]
        ai = game_data["ai_player"]

        # Player makes winning moves
        # X | X | X
        # O | O | -
        # - | - | -
        game.make_move(0, 0, "X")  # Player
        ai_move = ai.get_move(game.get_board())
        game.make_move(ai_move[0], ai_move[1], "O")  # AI

        game.make_move(0, 1, "X")  # Player
        ai_move = ai.get_move(game.get_board())
        game.make_move(ai_move[0], ai_move[1], "O")  # AI

        game.make_move(0, 2, "X")  # Player wins

        assert game.check_winner() == "X"
        assert game.is_game_over()

        # Update stats
        manager.update_stats(user_id, "win")
        stats = manager.get_stats(user_id)
        assert stats["wins"] == 1
        assert stats["total_games"] == 1

    def test_complete_single_player_game_draw(self):
        """Test complete single-player game ending in draw."""
        manager = GameManager()
        user_id = 12346

        game_data = manager.create_single_player_game(user_id, "X")
        game = game_data["game"]

        # Force a draw scenario
        # X | O | X
        # X | O | O
        # O | X | X
        moves = [
            (0, 0, "X"),
            (0, 1, "O"),
            (0, 2, "X"),
            (1, 0, "X"),
            (1, 1, "O"),
            (2, 1, "X"),
            (1, 2, "O"),
            (2, 2, "X"),
            (2, 0, "O"),
        ]

        for row, col, symbol in moves:
            game.make_move(row, col, symbol)

        assert game.is_draw()
        assert game.is_game_over()
        assert game.check_winner() is None

        manager.update_stats(user_id, "draw")
        stats = manager.get_stats(user_id)
        assert stats["draws"] == 1

    def test_single_player_game_lifecycle(self):
        """Test complete lifecycle of a single-player game."""
        manager = GameManager()
        user_id = 12347

        # Create game
        game_data = manager.create_single_player_game(user_id, "O")
        assert manager.get_game(user_id) is not None

        # Play some moves
        game = game_data["game"]
        ai = game_data["ai_player"]

        # AI goes first (X)
        ai_move = ai.get_move(game.get_board())
        game.make_move(ai_move[0], ai_move[1], "X")

        # Player moves
        game.make_move(1, 1, "O")

        # Delete game
        manager.delete_game(user_id)
        assert manager.get_game(user_id) is None

        # Stats should still exist
        stats = manager.get_stats(user_id)
        assert stats is not None


class TestMultiplayerIntegration:
    """Integration tests for multiplayer game flow."""

    def test_complete_multiplayer_game_flow(self):
        """Test complete multiplayer game from creation to completion."""
        manager = GameManager()
        creator_id = 20001
        joiner_id = 20002

        # Creator creates game
        game_code = manager.create_multiplayer_game(creator_id, "X")
        assert game_code is not None
        assert len(game_code) == GameSettings.GAME_CODE_LENGTH

        # Verify game exists
        game_data = manager.get_game_by_code(game_code)
        assert game_data is not None
        assert game_data["creator_id"] == creator_id
        assert not game_data["is_active"]

        # Joiner joins game
        success = manager.join_multiplayer_game(joiner_id, game_code)
        assert success

        # Verify game is now active
        game_data = manager.get_game_by_code(game_code)
        assert game_data["is_active"]
        assert game_data["joiner_id"] == joiner_id

        # Play game to completion
        game = game_data["game"]

        # X wins (creator)
        # X | X | X
        # O | O | -
        # - | - | -
        game.make_move(0, 0, "X")  # Creator
        manager.switch_turn(creator_id)

        game.make_move(1, 0, "O")  # Joiner
        manager.switch_turn(joiner_id)

        game.make_move(0, 1, "X")  # Creator
        manager.switch_turn(creator_id)

        game.make_move(1, 1, "O")  # Joiner
        manager.switch_turn(joiner_id)

        game.make_move(0, 2, "X")  # Creator wins

        assert game.check_winner() == "X"

        # Update stats
        manager.update_stats(creator_id, "win")
        manager.update_stats(joiner_id, "loss")

        creator_stats = manager.get_stats(creator_id)
        joiner_stats = manager.get_stats(joiner_id)

        assert creator_stats["wins"] == 1
        assert joiner_stats["losses"] == 1

        # Clean up
        manager.delete_multiplayer_game(creator_id)

    def test_multiplayer_turn_management(self):
        """Test turn management in multiplayer games."""
        manager = GameManager()
        creator_id = 20003
        joiner_id = 20004

        game_code = manager.create_multiplayer_game(creator_id, "X")
        manager.join_multiplayer_game(joiner_id, game_code)

        # X (creator) should go first
        assert manager.is_player_turn(creator_id)
        assert not manager.is_player_turn(joiner_id)

        # Switch turn
        manager.switch_turn(creator_id)
        assert not manager.is_player_turn(creator_id)
        assert manager.is_player_turn(joiner_id)

        # Switch back
        manager.switch_turn(joiner_id)
        assert manager.is_player_turn(creator_id)
        assert not manager.is_player_turn(joiner_id)

    def test_multiplayer_opponent_identification(self):
        """Test opponent identification in multiplayer games."""
        manager = GameManager()
        creator_id = 20005
        joiner_id = 20006

        game_code = manager.create_multiplayer_game(creator_id, "O")
        manager.join_multiplayer_game(joiner_id, game_code)

        # Get opponents
        creator_opponent = manager.get_opponent_id(creator_id)
        joiner_opponent = manager.get_opponent_id(joiner_id)

        assert creator_opponent == joiner_id
        assert joiner_opponent == creator_id


class TestStatsPersistence:
    """Integration tests for stats persistence across games."""

    def test_stats_persist_across_multiple_games(self):
        """Test that stats accumulate correctly across multiple games."""
        manager = GameManager()
        user_id = 30001

        # Play multiple games
        for i in range(3):
            game_data = manager.create_single_player_game(user_id, "X")
            manager.delete_game(user_id)

        # Record various results
        manager.update_stats(user_id, "win")
        manager.update_stats(user_id, "win")
        manager.update_stats(user_id, "loss")
        manager.update_stats(user_id, "draw")

        stats = manager.get_stats(user_id)
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["draws"] == 1
        assert stats["total_games"] == 4

    def test_stats_independent_per_user(self):
        """Test that stats are independent for each user."""
        manager = GameManager()
        user1_id = 30002
        user2_id = 30003

        # User 1 plays and wins
        manager.create_single_player_game(user1_id, "X")
        manager.update_stats(user1_id, "win")
        manager.delete_game(user1_id)

        # User 2 plays and loses
        manager.create_single_player_game(user2_id, "O")
        manager.update_stats(user2_id, "loss")
        manager.delete_game(user2_id)

        # Verify independent stats
        user1_stats = manager.get_stats(user1_id)
        user2_stats = manager.get_stats(user2_id)

        assert user1_stats["wins"] == 1
        assert user1_stats["losses"] == 0
        assert user2_stats["wins"] == 0
        assert user2_stats["losses"] == 1

    def test_stats_survive_game_deletion(self):
        """Test that stats persist after game deletion."""
        manager = GameManager()
        user_id = 30004

        # Create and delete multiple games
        for _ in range(5):
            manager.create_single_player_game(user_id, "X")
            manager.update_stats(user_id, "win")
            manager.delete_game(user_id)

        # Stats should still be there
        stats = manager.get_stats(user_id)
        assert stats["wins"] == 5
        assert stats["total_games"] == 5


class TestErrorRecovery:
    """Integration tests for error recovery scenarios."""

    def test_recovery_from_invalid_move(self):
        """Test game continues after invalid move attempt."""
        manager = GameManager()
        user_id = 40001

        game_data = manager.create_single_player_game(user_id, "X")
        game = game_data["game"]

        # Make valid move
        game.make_move(0, 0, "X")

        # Try invalid move (occupied cell)
        with pytest.raises(ValueError):
            game.make_move(0, 0, "X")

        # Game should still be playable
        game.make_move(0, 1, "X")
        assert game.get_board()[0][1] == "X"

    def test_recovery_from_invalid_game_code(self):
        """Test handling of invalid game code."""
        manager = GameManager()
        user_id = 40002

        # Try to join with invalid code
        result = manager.join_multiplayer_game(user_id, "INVALID")
        assert result is False

        # User should still be able to create their own game
        game_code = manager.create_multiplayer_game(user_id, "X")
        assert game_code is not None

    def test_recovery_from_concurrent_game_attempt(self):
        """Test handling when user tries to create multiple games."""
        manager = GameManager()
        user_id = 40003

        # Create first game
        game_data1 = manager.create_single_player_game(user_id, "X")
        assert game_data1 is not None

        # Try to create second game (should fail or replace)
        game_data2 = manager.create_single_player_game(user_id, "O")
        assert game_data2 is not None

        # Should only have one active game
        active_game = manager.get_game(user_id)
        assert active_game is not None

    def test_game_state_consistency_after_errors(self):
        """Test that game state remains consistent after errors."""
        manager = GameManager()
        user_id = 40004

        game_data = manager.create_single_player_game(user_id, "X")
        game = game_data["game"]

        # Make some valid moves
        game.make_move(0, 0, "X")
        game.make_move(1, 1, "O")

        # Try invalid moves
        try:
            game.make_move(0, 0, "X")  # Occupied
        except ValueError:
            pass

        try:
            game.make_move(5, 5, "X")  # Out of bounds
        except ValueError:
            pass

        # Verify game state is still consistent
        board = game.get_board()
        assert board[0][0] == "X"
        assert board[1][1] == "O"
        assert not game.is_game_over()


class TestConcurrentScenarios:
    """Integration tests for concurrent user scenarios."""

    def test_multiple_users_simultaneous_games(self):
        """Test multiple users playing simultaneously."""
        manager = GameManager()
        users = [50001, 50002, 50003, 50004, 50005]

        # Create games for all users
        for user_id in users:
            game_data = manager.create_single_player_game(user_id, "X")
            assert game_data is not None

        # Verify all games exist independently
        for user_id in users:
            game_data = manager.get_game(user_id)
            assert game_data is not None
            assert game_data["player_symbol"] == "X"

        # Each user makes moves independently
        for user_id in users:
            game_data = manager.get_game(user_id)
            game = game_data["game"]
            game.make_move(0, 0, "X")

        # Verify moves are independent
        for user_id in users:
            game_data = manager.get_game(user_id)
            game = game_data["game"]
            board = game.get_board()
            assert board[0][0] == "X"

    def test_concurrent_multiplayer_games(self):
        """Test multiple concurrent multiplayer games."""
        manager = GameManager()

        # Create multiple multiplayer games
        games = []
        for i in range(3):
            creator_id = 50010 + i * 2
            joiner_id = 50011 + i * 2

            game_code = manager.create_multiplayer_game(creator_id, "X")
            manager.join_multiplayer_game(joiner_id, game_code)

            games.append(
                {
                    "code": game_code,
                    "creator": creator_id,
                    "joiner": joiner_id,
                }
            )

        # Verify all games are independent
        for game_info in games:
            game_data = manager.get_game_by_code(game_info["code"])
            assert game_data is not None
            assert game_data["creator_id"] == game_info["creator"]
            assert game_data["joiner_id"] == game_info["joiner"]

    def test_unique_game_codes_generation(self):
        """Test that generated game codes are unique."""
        manager = GameManager()
        codes = set()

        # Generate multiple game codes
        for i in range(20):
            user_id = 50020 + i
            game_code = manager.create_multiplayer_game(user_id, "X")
            codes.add(game_code)
            manager.delete_multiplayer_game(user_id)

        # All codes should be unique
        assert len(codes) == 20


class TestCompleteGameScenarios:
    """Integration tests for complete realistic game scenarios."""

    def test_realistic_single_player_session(self):
        """Test a realistic single-player session with multiple games."""
        manager = GameManager()
        user_id = 60001

        # User plays 3 games
        results = ["win", "loss", "draw"]

        for result in results:
            # Start game
            game_data = manager.create_single_player_game(user_id, "X")
            game = game_data["game"]

            # Play some moves (simplified)
            game.make_move(0, 0, "X")
            game.make_move(1, 1, "O")

            # Update stats
            manager.update_stats(user_id, result)

            # End game
            manager.delete_game(user_id)

        # Check final stats
        stats = manager.get_stats(user_id)
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["draws"] == 1
        assert stats["total_games"] == 3

    def test_realistic_multiplayer_session(self):
        """Test a realistic multiplayer session."""
        manager = GameManager()
        player1_id = 60002
        player2_id = 60003

        # Player 1 creates game
        game_code = manager.create_multiplayer_game(player1_id, "X")

        # Player 2 joins
        success = manager.join_multiplayer_game(player2_id, game_code)
        assert success

        # Get game
        game_data = manager.get_game_by_code(game_code)
        game = game_data["game"]

        # Play complete game
        moves = [
            (0, 0, "X", player1_id),  # Player 1
            (1, 0, "O", player2_id),  # Player 2
            (0, 1, "X", player1_id),  # Player 1
            (1, 1, "O", player2_id),  # Player 2
            (0, 2, "X", player1_id),  # Player 1 wins
        ]

        for row, col, symbol, player_id in moves:
            assert manager.is_player_turn(player_id)
            game.make_move(row, col, symbol)
            if not game.is_game_over():
                manager.switch_turn(player_id)

        # Verify winner
        assert game.check_winner() == "X"

        # Update stats
        manager.update_stats(player1_id, "win")
        manager.update_stats(player2_id, "loss")

        # Clean up
        manager.delete_multiplayer_game(player1_id)

        # Verify stats
        p1_stats = manager.get_stats(player1_id)
        p2_stats = manager.get_stats(player2_id)

        assert p1_stats["wins"] == 1
        assert p2_stats["losses"] == 1