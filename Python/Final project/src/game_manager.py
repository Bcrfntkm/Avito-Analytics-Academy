"""
Game Manager for Tic-Tac-Toe bot.

This module manages game state, player sessions, and statistics
for the Telegram bot.
"""

import random
import string
from typing import Dict, Optional

from src.ai_player import AIPlayer
from src.game import TicTacToeGame


class GameManager:
    """
    Manages game sessions and statistics for multiple users.

    This class handles creating, retrieving, and deleting games,
    as well as tracking player statistics.

    Attributes:
        games (Dict[int, Dict]): Dictionary mapping user_id to game data
    """

    def __init__(self) -> None:
        """Initialize the GameManager with empty game storage."""
        self.games: Dict[int, Dict] = {}
        self.multiplayer_games: Dict[str, Dict] = {}

    def create_single_player_game(self, user_id: int, player_symbol: str) -> None:
        """
        Create a new single-player game for a user.

        Args:
            user_id (int): The Telegram user ID
            player_symbol (str): The symbol chosen by the player ('X' or 'O')

        Raises:
            ValueError: If player_symbol is not 'X' or 'O'
        """
        if player_symbol not in ["X", "O"]:
            raise ValueError(
                f"Invalid player symbol: {player_symbol}. Must be 'X' or 'O'"
            )

        # Determine AI symbol (opposite of player)
        ai_symbol = "O" if player_symbol == "X" else "X"

        # Create new game instance
        game = TicTacToeGame()

        # Initialize AI player
        ai_player = AIPlayer(difficulty="random")

        # Store game data
        self.games[user_id] = {
            "game": game,
            "mode": "single",
            "player_symbol": player_symbol,
            "ai_symbol": ai_symbol,
            "ai_player": ai_player,
            "stats": self._get_or_create_stats(user_id),
        }

    def get_game(self, user_id: int) -> Optional[Dict]:
        """
        Retrieve the game data for a specific user.

        Args:
            user_id (int): The Telegram user ID

        Returns:
            Optional[Dict]: Game data dictionary if exists, None otherwise.
                Dictionary contains:
                - game: TicTacToeGame instance
                - mode: 'single' or 'multi'
                - player_symbol: User's symbol ('X' or 'O')
                - ai_symbol: AI's symbol (for single-player)
                - ai_player: AIPlayer instance (for single-player)
                - stats: Statistics dictionary
        """
        return self.games.get(user_id)

    def delete_game(self, user_id: int) -> None:
        """
        Delete a user's active game.

        Args:
            user_id (int): The Telegram user ID
        """
        if user_id in self.games:
            # Preserve stats before deleting
            stats = self.games[user_id].get("stats", {})
            del self.games[user_id]
            # Store stats separately for future games
            if user_id not in self._stats_storage:
                self._stats_storage[user_id] = stats

    def update_stats(self, user_id: int, result: str) -> None:
        """
        Update statistics for a user based on game result.

        Args:
            user_id (int): The Telegram user ID
            result (str): The game result ('win', 'loss', or 'draw')

        Raises:
            ValueError: If result is not 'win', 'loss', or 'draw'
        """
        valid_results = ["win", "loss", "draw"]
        if result not in valid_results:
            raise ValueError(
                f"Invalid result: {result}. Must be one of {valid_results}"
            )

        # Get or create stats
        stats = self._get_or_create_stats(user_id)

        # Update the appropriate counter
        if result == "win":
            stats["wins"] += 1
        elif result == "loss":
            stats["losses"] += 1
        elif result == "draw":
            stats["draws"] += 1

        # Update stats in active game if exists
        if user_id in self.games:
            self.games[user_id]["stats"] = stats

        # Also update in persistent storage
        self._stats_storage[user_id] = stats

    def get_stats(self, user_id: int) -> Dict[str, int]:
        """
        Get statistics for a user.

        Args:
            user_id (int): The Telegram user ID

        Returns:
            Dict[str, int]: Dictionary with 'wins', 'losses', and 'draws' counts
        """
        return self._get_or_create_stats(user_id).copy()

    def _get_or_create_stats(self, user_id: int) -> Dict[str, int]:
        """
        Get existing stats or create new ones for a user.

        Args:
            user_id (int): The Telegram user ID

        Returns:
            Dict[str, int]: Statistics dictionary
        """
        if not hasattr(self, "_stats_storage"):
            self._stats_storage: Dict[int, Dict[str, int]] = {}

        if user_id not in self._stats_storage:
            self._stats_storage[user_id] = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
            }

        return self._stats_storage[user_id]

    def generate_game_code(self) -> str:
        """
        Generate a unique 6-character alphanumeric game code.

        Returns:
            str: A unique 6-character game code (uppercase letters and digits)
        """
        while True:
            code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if code not in self.multiplayer_games:
                return code

    def create_multiplayer_game(
        self, creator_id: int, creator_symbol: str, game_code: str
    ) -> str:
        """
        Create a new multiplayer game and return game code.

        Args:
            creator_id (int): User ID of game creator
            creator_symbol (str): Symbol chosen by creator ('X' or 'O')
            game_code (str): Unique game code for joining

        Returns:
            str: The game code for joining

        Raises:
            ValueError: If creator_symbol is not 'X' or 'O'
        """
        if creator_symbol not in ["X", "O"]:
            raise ValueError(
                f"Invalid creator symbol: {creator_symbol}. Must be 'X' or 'O'"
            )

        # Determine joiner symbol (opposite of creator)
        joiner_symbol = "O" if creator_symbol == "X" else "X"

        # Create new game instance
        game = TicTacToeGame()

        # Store multiplayer game data
        self.multiplayer_games[game_code] = {
            "game": game,
            "mode": "multi",
            "creator_id": creator_id,
            "joiner_id": None,
            "creator_symbol": creator_symbol,
            "joiner_symbol": joiner_symbol,
            "current_turn": creator_id if creator_symbol == "X" else None,
            "game_code": game_code,
            "status": "waiting",
        }

        # Also store reference in user's games
        self.games[creator_id] = self.multiplayer_games[game_code]

        return game_code

    def join_multiplayer_game(self, joiner_id: int, game_code: str) -> bool:
        """
        Join existing game by code.

        Args:
            joiner_id (int): User ID of player joining
            game_code (str): Game code to join

        Returns:
            bool: True if successfully joined, False otherwise
        """
        if game_code not in self.multiplayer_games:
            return False

        game_data = self.multiplayer_games[game_code]

        # Check if game is still waiting for a player
        if game_data["status"] != "waiting":
            return False

        # Check if joiner is not the creator
        if game_data["creator_id"] == joiner_id:
            return False

        # Join the game
        game_data["joiner_id"] = joiner_id
        game_data["status"] = "active"

        # Set current turn to joiner if they are X
        if game_data["joiner_symbol"] == "X":
            game_data["current_turn"] = joiner_id

        # Store reference in joiner's games
        self.games[joiner_id] = game_data

        return True

    def get_game_by_code(self, game_code: str) -> Optional[Dict]:
        """
        Get game by its code.

        Args:
            game_code (str): The game code

        Returns:
            Optional[Dict]: Game data dictionary if exists, None otherwise
        """
        return self.multiplayer_games.get(game_code)

    def get_multiplayer_game(self, user_id: int) -> Optional[Dict]:
        """
        Get multiplayer game where user is participant.

        Args:
            user_id (int): The Telegram user ID

        Returns:
            Optional[Dict]: Game data dictionary if exists and is multiplayer,
                None otherwise
        """
        game_data = self.games.get(user_id)
        if game_data and game_data.get("mode") == "multi":
            return game_data
        return None

    def is_player_turn(self, user_id: int, game_code: str) -> bool:
        """
        Check if it's the user's turn.

        Args:
            user_id (int): The Telegram user ID
            game_code (str): The game code

        Returns:
            bool: True if it's the user's turn, False otherwise
        """
        game_data = self.get_game_by_code(game_code)
        if not game_data:
            return False

        return game_data.get("current_turn") == user_id

    def get_opponent_id(self, user_id: int, game_code: str) -> Optional[int]:
        """
        Get opponent's user ID.

        Args:
            user_id (int): The Telegram user ID
            game_code (str): The game code

        Returns:
            Optional[int]: Opponent's user ID if found, None otherwise
        """
        game_data = self.get_game_by_code(game_code)
        if not game_data:
            return None

        creator_id = game_data.get("creator_id")
        joiner_id = game_data.get("joiner_id")

        if user_id == creator_id:
            return joiner_id
        elif user_id == joiner_id:
            return creator_id

        return None

    def switch_turn(self, game_code: str) -> None:
        """
        Switch turn to the other player.

        Args:
            game_code (str): The game code
        """
        game_data = self.get_game_by_code(game_code)
        if not game_data:
            return

        current_turn = game_data.get("current_turn")
        creator_id = game_data.get("creator_id")
        joiner_id = game_data.get("joiner_id")

        if current_turn == creator_id:
            game_data["current_turn"] = joiner_id
        else:
            game_data["current_turn"] = creator_id

    def get_player_symbol(self, user_id: int, game_code: str) -> Optional[str]:
        """
        Get the symbol for a specific player in a multiplayer game.

        Args:
            user_id (int): The Telegram user ID
            game_code (str): The game code

        Returns:
            Optional[str]: Player's symbol ('X' or 'O') if found, None otherwise
        """
        game_data = self.get_game_by_code(game_code)
        if not game_data:
            return None

        if user_id == game_data.get("creator_id"):
            return game_data.get("creator_symbol")
        elif user_id == game_data.get("joiner_id"):
            return game_data.get("joiner_symbol")

        return None

    def delete_multiplayer_game(self, game_code: str) -> None:
        """
        Delete a multiplayer game and clean up references.

        Args:
            game_code (str): The game code to delete
        """
        if game_code not in self.multiplayer_games:
            return

        game_data = self.multiplayer_games[game_code]
        creator_id = game_data.get("creator_id")
        joiner_id = game_data.get("joiner_id")

        # Remove from user games
        if creator_id and creator_id in self.games:
            if self.games[creator_id].get("game_code") == game_code:
                del self.games[creator_id]

        if joiner_id and joiner_id in self.games:
            if self.games[joiner_id].get("game_code") == game_code:
                del self.games[joiner_id]

        # Remove from multiplayer games
        del self.multiplayer_games[game_code]
