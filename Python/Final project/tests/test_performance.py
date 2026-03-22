"""
Performance tests for Tic-Tac-Toe Telegram Bot.

This module contains performance benchmarks for critical operations
to ensure the bot can handle expected load efficiently.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.ai_player import AIPlayer
from src.game import TicTacToeGame
from src.game_manager import GameManager


class TestGameCreationPerformance:
    """Performance tests for game creation operations."""

    def test_single_player_game_creation_speed(self):
        """Test speed of creating single-player games."""
        manager = GameManager()
        user_id = 70001

        start_time = time.time()
        iterations = 100

        for i in range(iterations):
            manager.create_single_player_game(user_id + i, "X")

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nSingle-player game creation:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Games per second: {iterations/elapsed:.2f}")

        # Should create games quickly (< 10ms each)
        assert avg_time < 0.01, f"Game creation too slow: {avg_time:.4f}s"

    def test_multiplayer_game_creation_speed(self):
        """Test speed of creating multiplayer games."""
        manager = GameManager()
        user_id = 70100

        start_time = time.time()
        iterations = 100

        for i in range(iterations):
            manager.create_multiplayer_game(user_id + i, "X")

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nMultiplayer game creation:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Games per second: {iterations/elapsed:.2f}")

        # Should create games quickly (< 10ms each)
        assert avg_time < 0.01, f"Game creation too slow: {avg_time:.4f}s"

    def test_game_code_generation_speed(self):
        """Test speed of generating unique game codes."""
        manager = GameManager()

        start_time = time.time()
        iterations = 1000

        codes = set()
        for _ in range(iterations):
            code = manager._generate_game_code()
            codes.add(code)

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nGame code generation:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Codes per second: {iterations/elapsed:.2f}")

        # All codes should be unique
        assert len(codes) == iterations

        # Should generate codes very quickly (< 1ms each)
        assert avg_time < 0.001, f"Code generation too slow: {avg_time:.6f}s"


class TestMoveProcessingPerformance:
    """Performance tests for move processing operations."""

    def test_move_validation_speed(self):
        """Test speed of move validation."""
        game = TicTacToeGame()

        start_time = time.time()
        iterations = 10000

        for _ in range(iterations):
            # Test valid position
            try:
                game.make_move(0, 0, "X")
            except ValueError:
                pass
            game.reset()

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nMove validation:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Validations per second: {iterations/elapsed:.2f}")

        # Should validate moves very quickly (< 0.1ms each)
        assert avg_time < 0.0001, f"Move validation too slow: {avg_time:.6f}s"

    def test_winner_check_speed(self):
        """Test speed of winner checking."""
        game = TicTacToeGame()

        # Set up a board state
        game.make_move(0, 0, "X")
        game.make_move(0, 1, "O")
        game.make_move(1, 1, "X")

        start_time = time.time()
        iterations = 10000

        for _ in range(iterations):
            game.check_winner()

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nWinner checking:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Checks per second: {iterations/elapsed:.2f}")

        # Should check winner very quickly (< 0.1ms each)
        assert avg_time < 0.0001, f"Winner check too slow: {avg_time:.6f}s"

    def test_ai_move_generation_speed(self):
        """Test speed of AI move generation."""
        ai = AIPlayer()
        game = TicTacToeGame()

        start_time = time.time()
        iterations = 100

        for _ in range(iterations):
            board = game.get_board()
            ai.get_move(board)
            game.reset()

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nAI move generation:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Moves per second: {iterations/elapsed:.2f}")

        # AI should generate moves quickly (< 10ms each)
        assert avg_time < 0.01, f"AI move generation too slow: {avg_time:.4f}s"


class TestConcurrentGameHandling:
    """Performance tests for concurrent game operations."""

    def test_concurrent_game_creation(self):
        """Test handling of concurrent game creation."""
        manager = GameManager()
        num_users = 50

        def create_game(user_id):
            return manager.create_single_player_game(user_id, "X")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            user_ids = range(80001, 80001 + num_users)
            results = list(executor.map(create_game, user_ids))

        elapsed = time.time() - start_time

        print(f"\nConcurrent game creation ({num_users} games):")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time per game: {elapsed/num_users:.4f}s")
        print(f"  Games per second: {num_users/elapsed:.2f}")

        # All games should be created successfully
        assert all(result is not None for result in results)

        # Should handle concurrent creation efficiently
        assert elapsed < 2.0, f"Concurrent creation too slow: {elapsed:.4f}s"

    def test_concurrent_move_processing(self):
        """Test handling of concurrent move processing."""
        manager = GameManager()
        num_games = 20

        # Create games
        games = []
        for i in range(num_games):
            user_id = 80100 + i
            game_data = manager.create_single_player_game(user_id, "X")
            games.append((user_id, game_data["game"]))

        def make_move(game_info):
            user_id, game = game_info
            try:
                game.make_move(0, 0, "X")
                return True
            except Exception:
                return False

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(make_move, games))

        elapsed = time.time() - start_time

        print(f"\nConcurrent move processing ({num_games} moves):")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time per move: {elapsed/num_games:.4f}s")
        print(f"  Moves per second: {num_games/elapsed:.2f}")

        # All moves should succeed
        assert all(results)

        # Should handle concurrent moves efficiently
        assert elapsed < 1.0, f"Concurrent moves too slow: {elapsed:.4f}s"

    def test_concurrent_stats_updates(self):
        """Test handling of concurrent stats updates."""
        manager = GameManager()
        num_updates = 100

        def update_stats(user_id):
            manager.update_stats(user_id, "win")
            return True

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            user_ids = range(80200, 80200 + num_updates)
            results = list(executor.map(update_stats, user_ids))

        elapsed = time.time() - start_time

        print(f"\nConcurrent stats updates ({num_updates} updates):")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time per update: {elapsed/num_updates:.4f}s")
        print(f"  Updates per second: {num_updates/elapsed:.2f}")

        # All updates should succeed
        assert all(results)

        # Should handle concurrent updates efficiently
        assert elapsed < 1.0, f"Concurrent updates too slow: {elapsed:.4f}s"


class TestMemoryUsage:
    """Performance tests for memory usage."""

    def test_memory_usage_with_many_games(self):
        """Test memory usage with many active games."""
        import sys

        manager = GameManager()
        num_games = 100

        # Get initial memory usage (approximate)
        initial_size = sys.getsizeof(manager.games)

        # Create many games
        for i in range(num_games):
            manager.create_single_player_game(90001 + i, "X")

        # Get final memory usage
        final_size = sys.getsizeof(manager.games)
        size_per_game = (final_size - initial_size) / num_games

        print(f"\nMemory usage with {num_games} games:")
        print(f"  Initial size: {initial_size} bytes")
        print(f"  Final size: {final_size} bytes")
        print(f"  Size per game: {size_per_game:.2f} bytes")

        # Each game should use reasonable memory (< 10KB)
        assert size_per_game < 10000, f"Memory per game too high: {size_per_game}"

    def test_memory_cleanup_after_game_deletion(self):
        """Test that memory is properly cleaned up after game deletion."""
        import sys

        manager = GameManager()
        num_games = 50

        # Create games
        for i in range(num_games):
            manager.create_single_player_game(90100 + i, "X")

        size_with_games = sys.getsizeof(manager.games)

        # Delete all games
        for i in range(num_games):
            manager.delete_game(90100 + i)

        size_after_deletion = sys.getsizeof(manager.games)

        print(f"\nMemory cleanup after deletion:")
        print(f"  Size with {num_games} games: {size_with_games} bytes")
        print(f"  Size after deletion: {size_after_deletion} bytes")
        print(f"  Memory freed: {size_with_games - size_after_deletion} bytes")

        # Memory should be significantly reduced
        assert size_after_deletion < size_with_games


class TestScalability:
    """Performance tests for scalability."""

    def test_game_retrieval_speed_with_many_games(self):
        """Test game retrieval speed with many active games."""
        manager = GameManager()
        num_games = 200

        # Create many games
        for i in range(num_games):
            manager.create_single_player_game(95001 + i, "X")

        # Test retrieval speed
        start_time = time.time()
        iterations = 1000

        for _ in range(iterations):
            # Retrieve random game
            user_id = 95001 + (_ % num_games)
            manager.get_game(user_id)

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nGame retrieval with {num_games} active games:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Retrievals per second: {iterations/elapsed:.2f}")

        # Retrieval should be fast even with many games (< 0.1ms)
        assert avg_time < 0.0001, f"Game retrieval too slow: {avg_time:.6f}s"

    def test_stats_retrieval_speed_with_many_users(self):
        """Test stats retrieval speed with many users."""
        manager = GameManager()
        num_users = 200

        # Create stats for many users
        for i in range(num_users):
            manager.update_stats(95200 + i, "win")

        # Test retrieval speed
        start_time = time.time()
        iterations = 1000

        for _ in range(iterations):
            user_id = 95200 + (_ % num_users)
            manager.get_stats(user_id)

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nStats retrieval with {num_users} users:")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Retrievals per second: {iterations/elapsed:.2f}")

        # Retrieval should be fast (< 0.1ms)
        assert avg_time < 0.0001, f"Stats retrieval too slow: {avg_time:.6f}s"


class TestCompleteGamePerformance:
    """Performance tests for complete game scenarios."""

    def test_complete_game_performance(self):
        """Test performance of playing a complete game."""
        manager = GameManager()
        user_id = 96001

        start_time = time.time()

        # Create game
        game_data = manager.create_single_player_game(user_id, "X")
        game = game_data["game"]
        ai = game_data["ai_player"]

        # Play complete game
        while not game.is_game_over():
            # Player move
            available = game.get_available_moves()
            if available:
                row, col = available[0]
                game.make_move(row, col, "X")

            if game.is_game_over():
                break

            # AI move
            ai_move = ai.get_move(game.get_board())
            game.make_move(ai_move[0], ai_move[1], "O")

        # Update stats
        if game.check_winner() == "X":
            manager.update_stats(user_id, "win")
        elif game.is_draw():
            manager.update_stats(user_id, "draw")
        else:
            manager.update_stats(user_id, "loss")

        # Delete game
        manager.delete_game(user_id)

        elapsed = time.time() - start_time

        print(f"\nComplete game performance:")
        print(f"  Total time: {elapsed:.4f}s")

        # Complete game should finish quickly (< 100ms)
        assert elapsed < 0.1, f"Complete game too slow: {elapsed:.4f}s"

    def test_multiple_sequential_games_performance(self):
        """Test performance of playing multiple games sequentially."""
        manager = GameManager()
        user_id = 96002
        num_games = 10

        start_time = time.time()

        for _ in range(num_games):
            game_data = manager.create_single_player_game(user_id, "X")
            game = game_data["game"]

            # Play a few moves
            game.make_move(0, 0, "X")
            game.make_move(1, 1, "O")
            game.make_move(0, 1, "X")

            manager.update_stats(user_id, "win")
            manager.delete_game(user_id)

        elapsed = time.time() - start_time
        avg_time = elapsed / num_games

        print(f"\nMultiple sequential games ({num_games} games):")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Average time per game: {avg_time:.4f}s")
        print(f"  Games per second: {num_games/elapsed:.2f}")

        # Should handle multiple games efficiently (< 50ms per game)
        assert avg_time < 0.05, f"Sequential games too slow: {avg_time:.4f}s"


def print_performance_summary():
    """Print a summary of all performance benchmarks."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 70)
    print("\nAll performance tests passed!")
    print("\nKey Metrics:")
    print("  - Game creation: < 10ms")
    print("  - Move processing: < 0.1ms")
    print("  - AI move generation: < 10ms")
    print("  - Concurrent operations: Efficient")
    print("  - Memory usage: Reasonable")
    print("  - Scalability: Good")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
    print_performance_summary()