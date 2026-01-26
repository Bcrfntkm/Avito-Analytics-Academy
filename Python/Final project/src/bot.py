"""
Tic-Tac-Toe Telegram Bot

A Telegram bot that allows users to play Tic-Tac-Toe game.
This module contains the main bot initialization and command handlers.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config.settings import ErrorMessages, ValidationRules
from src.error_handlers import (
    GameNotFoundError,
    InvalidGameCodeError,
    handle_game_errors,
    handle_telegram_errors,
    retry_on_error,
)
from src.game_manager import GameManager
from src.logging_config import (
    get_logger,
    log_game_event,
    log_user_action,
    setup_logging,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize game manager
game_manager = GameManager()


@handle_telegram_errors
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /start command.

    Sends a welcome message to the user when they start the bot.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.

    Raises:
        TelegramError: If message sending fails
    """
    user = update.effective_user
    log_user_action("start_command", user.id)

    welcome_message = (
        f"👋 Hello {user.mention_html()}!\n\n"
        "Welcome to Tic-Tac-Toe Bot! 🎮\n\n"
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show help information\n"
        "/newgame - Start a new game\n"
        "/stats - View your statistics"
    )
    await update.message.reply_html(welcome_message)
    logger.info(f"User {user.id} started the bot")


@handle_telegram_errors
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /help command.

    Provides detailed information about bot commands and gameplay.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.

    Raises:
        TelegramError: If message sending fails
    """
    user_id = update.effective_user.id
    log_user_action("help_command", user_id)

    help_text = (
        "🎮 <b>Tic-Tac-Toe Bot Help</b>\n\n"
        "<b>Commands:</b>\n"
        "/start - Start the bot and see welcome message\n"
        "/help - Show this help message\n"
        "/newgame - Start a new Tic-Tac-Toe game\n"
        "/stats - View your game statistics\n\n"
        "<b>How to play:</b>\n"
        "1. Use /newgame to start a new game\n"
        "2. Choose single-player mode to play against AI\n"
        "3. Select your symbol (X or O)\n"
        "4. Click on the buttons to make your move\n"
        "5. Try to get three in a row to win!\n\n"
        "Good luck! 🍀"
    )
    await update.message.reply_html(help_text)
    logger.info(f"User {user_id} requested help")


@handle_telegram_errors
@handle_game_errors
async def newgame_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /newgame command.

    Starts a new Tic-Tac-Toe game for the user.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.

    Raises:
        TelegramError: If message sending fails
    """
    user_id = update.effective_user.id
    log_user_action("newgame_command", user_id)

    # Check if user already has an active game
    existing_game = game_manager.get_game(user_id)
    if existing_game:
        logger.info(f"User {user_id} has existing game, deleting it")
        await update.message.reply_text(
            "⚠️ You already have an active game!\n"
            "Finish it first or use /newgame to start over."
        )
        # Delete the old game to allow starting a new one
        game_manager.delete_game(user_id)

    # Show mode selection keyboard
    keyboard = [
        [
            InlineKeyboardButton(
                "🤖 Single Player (vs AI)", callback_data="mode_single"
            ),
        ],
        [
            InlineKeyboardButton("👥 Multiplayer", callback_data="mode_multi"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🎮 <b>New Game</b>\n\n" "Choose game mode:",
        reply_markup=reply_markup,
        parse_mode="HTML",
    )
    logger.info(f"User {user_id} initiated new game")


@handle_telegram_errors
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /stats command.

    Displays the user's game statistics.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.

    Raises:
        TelegramError: If message sending fails
    """
    user_id = update.effective_user.id
    log_user_action("stats_command", user_id)

    try:
        stats = game_manager.get_stats(user_id)
        total_games = stats["wins"] + stats["losses"] + stats["draws"]
        win_rate = (stats["wins"] / total_games * 100) if total_games > 0 else 0

        stats_text = (
            "📊 <b>Your Statistics</b>\n\n"
            f"🎮 Total Games: {total_games}\n"
            f"🏆 Wins: {stats['wins']}\n"
            f"😔 Losses: {stats['losses']}\n"
            f"🤝 Draws: {stats['draws']}\n"
            f"📈 Win Rate: {win_rate:.1f}%"
        )

        await update.message.reply_html(stats_text)
        logger.info(f"User {user_id} requested statistics")
    except Exception as e:
        logger.error(f"Error retrieving stats for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(ErrorMessages.GENERIC_ERROR)


async def mode_selection_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle mode selection callback.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    mode = query.data.split("_")[1]  # Extract 'single' or 'multi' from callback_data

    if mode == "multi":
        # Show multiplayer options
        keyboard = [
            [
                InlineKeyboardButton("🎮 Create Game", callback_data="multi_create"),
                InlineKeyboardButton("🔗 Join Game", callback_data="multi_join"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "👥 <b>Multiplayer Mode</b>\n\n"
            "Create a new game or join an existing one:",
            reply_markup=reply_markup,
            parse_mode="HTML",
        )
        return

    # Show symbol selection for single-player
    keyboard = [
        [
            InlineKeyboardButton("❌ Play as X (go first)", callback_data="symbol_X"),
            InlineKeyboardButton("⭕ Play as O (go second)", callback_data="symbol_O"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🎮 <b>Single Player Mode</b>\n\n" "Choose your symbol:",
        reply_markup=reply_markup,
        parse_mode="HTML",
    )
    logger.info(f"User {user_id} selected {mode} mode")


async def symbol_selection_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle symbol selection callback and start the game.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    symbol = query.data.split("_")[1]  # Extract 'X' or 'O' from callback_data

    # Create the game
    game_manager.create_single_player_game(user_id, symbol)
    game_data = game_manager.get_game(user_id)

    # If player chose O, AI (X) goes first
    if symbol == "O":
        ai_player = game_data["ai_player"]
        game = game_data["game"]
        ai_move = ai_player.get_move(game)
        game.make_move(ai_move[0], ai_move[1], "X")

        await query.edit_message_text(
            f"🎮 <b>Game Started!</b>\n\n"
            f"You are playing as <b>{symbol}</b>\n"
            f"AI made the first move!\n\n"
            f"{_format_board_text(game)}",
            parse_mode="HTML",
        )
    else:
        await query.edit_message_text(
            f"🎮 <b>Game Started!</b>\n\n"
            f"You are playing as <b>{symbol}</b>\n"
            f"Your turn!\n\n"
            f"{_format_board_text(game_data['game'])}",
            parse_mode="HTML",
        )

    # Show the game board
    await _send_game_board(query.message, user_id)
    logger.info(f"User {user_id} started game as {symbol}")


async def board_button_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle board button clicks (player moves).

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    game_data = game_manager.get_game(user_id)

    if not game_data:
        await query.edit_message_text("❌ No active game found. Use /newgame to start!")
        return

    game = game_data["game"]
    player_symbol = game_data["player_symbol"]
    ai_symbol = game_data["ai_symbol"]

    # Parse position from callback_data (format: "pos_0" to "pos_8")
    position = int(query.data.split("_")[1])
    row = position // 3
    col = position % 3

    # Validate and make player move
    if not game.make_move(row, col, player_symbol):
        await query.answer("❌ Invalid move! Cell is already occupied.", show_alert=True)
        return

    # Check if game is over after player move
    winner = game.check_winner()
    if winner:
        result = "win" if winner == player_symbol else "loss"
        game_manager.update_stats(user_id, result)
        await _send_game_over_message(query.message, game, user_id, winner)
        game_manager.delete_game(user_id)
        return

    if game.is_draw():
        game_manager.update_stats(user_id, "draw")
        await _send_game_over_message(query.message, game, user_id, None)
        game_manager.delete_game(user_id)
        return

    # AI makes move
    ai_player = game_data["ai_player"]
    try:
        ai_move = ai_player.get_move(game)
        game.make_move(ai_move[0], ai_move[1], ai_symbol)
    except ValueError:
        # No moves available (shouldn't happen, but handle gracefully)
        await query.edit_message_text("❌ Error: No moves available for AI")
        return

    # Check if game is over after AI move
    winner = game.check_winner()
    if winner:
        result = "win" if winner == player_symbol else "loss"
        game_manager.update_stats(user_id, result)
        await _send_game_over_message(query.message, game, user_id, winner)
        game_manager.delete_game(user_id)
        return

    if game.is_draw():
        game_manager.update_stats(user_id, "draw")
        await _send_game_over_message(query.message, game, user_id, None)
        game_manager.delete_game(user_id)
        return

    # Game continues - update board
    await _send_game_board(query.message, user_id)


async def multi_create_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle multiplayer game creation.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Check if user already has an active game
    existing_game = game_manager.get_game(user_id)
    if existing_game:
        await query.edit_message_text(
            "⚠️ You already have an active game!\n"
            "Finish it first or use /newgame to start over."
        )
        return

    # Show symbol selection
    keyboard = [
        [
            InlineKeyboardButton(
                "❌ Play as X (go first)", callback_data="multi_symbol_X"
            ),
            InlineKeyboardButton(
                "⭕ Play as O (go second)", callback_data="multi_symbol_O"
            ),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🎮 <b>Create Multiplayer Game</b>\n\n" "Choose your symbol:",
        reply_markup=reply_markup,
        parse_mode="HTML",
    )
    logger.info(f"User {user_id} initiated multiplayer game creation")


async def multi_join_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle multiplayer game join request.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    # Check if user already has an active game
    existing_game = game_manager.get_game(user_id)
    if existing_game:
        await query.edit_message_text(
            "⚠️ You already have an active game!\n"
            "Finish it first or use /newgame to start over."
        )
        return

    # Store that user wants to join a game
    context.user_data["awaiting_game_code"] = True

    await query.edit_message_text(
        "🔗 <b>Join Multiplayer Game</b>\n\n" "Please enter the 6-character game code:",
        parse_mode="HTML",
    )
    logger.info(f"User {user_id} requested to join multiplayer game")


async def multi_symbol_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle symbol selection for multiplayer game creation.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    symbol = query.data.split("_")[2]  # Extract 'X' or 'O'

    # Generate unique game code
    game_code = game_manager.generate_game_code()

    # Create multiplayer game
    game_manager.create_multiplayer_game(user_id, symbol, game_code)

    # Show waiting message with cancel button
    keyboard = [
        [InlineKeyboardButton("❌ Cancel Game", callback_data=f"cancel_{game_code}")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        f"🎮 <b>Game Created!</b>\n\n"
        f"Your symbol: <b>{symbol}</b>\n"
        f"Game code: <code>{game_code}</code>\n\n"
        f"⏳ Waiting for opponent to join...\n\n"
        f"Share this code with your friend!",
        reply_markup=reply_markup,
        parse_mode="HTML",
    )
    logger.info(f"User {user_id} created multiplayer game {game_code} as {symbol}")


async def cancel_game_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle game cancellation.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    game_code = query.data.split("_")[1]

    # Verify user is the creator
    game_data = game_manager.get_game_by_code(game_code)
    if not game_data or game_data.get("creator_id") != user_id:
        await query.edit_message_text("❌ Cannot cancel this game.")
        return

    # Delete the game
    game_manager.delete_multiplayer_game(game_code)

    await query.edit_message_text(
        "❌ <b>Game Cancelled</b>\n\n" "Use /newgame to start a new game.",
        parse_mode="HTML",
    )
    logger.info(f"User {user_id} cancelled game {game_code}")


@handle_telegram_errors
@handle_game_errors
async def game_code_message_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle game code input from user.

    Validates the game code format and attempts to join the multiplayer game.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.

    Raises:
        InvalidGameCodeError: If game code format is invalid
        GameNotFoundError: If game code doesn't exist
        TelegramError: If message sending fails
    """
    # Check if user is awaiting game code
    if not context.user_data.get("awaiting_game_code"):
        return

    user_id = update.effective_user.id
    game_code = update.message.text.strip().upper()

    # Clear the awaiting flag
    context.user_data["awaiting_game_code"] = False

    log_user_action("join_game_attempt", user_id, game_code=game_code)

    # Validate game code format
    if not ValidationRules.validate_game_code(game_code):
        logger.info(f"Invalid game code format from user {user_id}: {game_code}")
        raise InvalidGameCodeError(f"Invalid game code format: {game_code}")

    # Try to join the game
    try:
        success = game_manager.join_multiplayer_game(user_id, game_code)
    except Exception as e:
        logger.error(
            f"Error joining game {game_code} for user {user_id}: {e}", exc_info=True
        )
        await update.message.reply_text(ErrorMessages.GENERIC_ERROR)
        return

    if not success:
        logger.info(f"User {user_id} failed to join game {game_code}")
        await update.message.reply_text(
            "❌ Could not join game!\n\n"
            "Possible reasons:\n"
            "• Game code doesn't exist\n"
            "• Game already started\n"
            "• You are the game creator\n\n"
            "Use /newgame to try again."
        )
        return

    # Get game data
    game_data = game_manager.get_game_by_code(game_code)
    creator_id = game_data["creator_id"]
    joiner_symbol = game_data["joiner_symbol"]
    creator_symbol = game_data["creator_symbol"]

    # Notify joiner
    await update.message.reply_text(
        f"✅ <b>Joined Game!</b>\n\n"
        f"Your symbol: <b>{joiner_symbol}</b>\n"
        f"Game code: <code>{game_code}</code>\n\n"
        f"Game is starting...",
        parse_mode="HTML",
    )

    # Notify creator
    try:
        await context.bot.send_message(
            chat_id=creator_id,
            text=f"✅ <b>Opponent Joined!</b>\n\n"
            f"Game code: <code>{game_code}</code>\n"
            f"Your symbol: <b>{creator_symbol}</b>\n"
            f"Opponent symbol: <b>{joiner_symbol}</b>\n\n"
            f"Game is starting...",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error(f"Failed to notify creator {creator_id}: {e}")

    # Send board to both players
    await _send_multiplayer_board(context, game_code)

    logger.info(f"User {user_id} joined game {game_code}")


async def multiplayer_board_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle board button clicks in multiplayer games.

    Args:
        update: The incoming update from Telegram.
        context: The context object for the handler.
    """
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    game_data = game_manager.get_multiplayer_game(user_id)

    if not game_data:
        await query.edit_message_text("❌ No active multiplayer game found.")
        return

    game_code = game_data["game_code"]
    game = game_data["game"]

    # Check if it's player's turn
    if not game_manager.is_player_turn(user_id, game_code):
        await query.answer("⏳ It's not your turn!", show_alert=True)
        return

    # Parse position from callback_data (format: "multi_pos_0" to "multi_pos_8")
    position = int(query.data.split("_")[2])
    row = position // 3
    col = position % 3

    # Get player symbol
    player_symbol = game_manager.get_player_symbol(user_id, game_code)

    # Validate and make move
    if not game.make_move(row, col, player_symbol):
        await query.answer("❌ Invalid move! Cell is already occupied.", show_alert=True)
        return

    # Check if game is over
    winner = game.check_winner()
    if winner:
        await _handle_multiplayer_game_over(context, game_code, winner)
        return

    if game.is_draw():
        await _handle_multiplayer_game_over(context, game_code, None)
        return

    # Switch turn
    game_manager.switch_turn(game_code)

    # Update board for both players
    await _send_multiplayer_board(context, game_code)


async def _send_multiplayer_board(
    context: ContextTypes.DEFAULT_TYPE, game_code: str
) -> None:
    """
    Send or update the multiplayer game board to both players.

    Args:
        context: The context object for the handler
        game_code: The game code
    """
    game_data = game_manager.get_game_by_code(game_code)
    if not game_data:
        return

    game = game_data["game"]
    board = game.get_board()
    creator_id = game_data["creator_id"]
    joiner_id = game_data["joiner_id"]
    current_turn = game_data["current_turn"]
    creator_symbol = game_data["creator_symbol"]
    joiner_symbol = game_data["joiner_symbol"]

    # Create inline keyboard for the board
    keyboard = []
    for row_idx in range(3):
        row_buttons = []
        for col_idx in range(3):
            position = row_idx * 3 + col_idx
            cell = board[row_idx][col_idx]

            # Show X, O, or position number
            if cell == "X":
                button_text = "❌"
            elif cell == "O":
                button_text = "⭕"
            else:
                button_text = str(position + 1)

            row_buttons.append(
                InlineKeyboardButton(
                    button_text,
                    callback_data=f"multi_pos_{position}",
                )
            )
        keyboard.append(row_buttons)

    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send to creator
    creator_turn = current_turn == creator_id
    turn_text = "Your turn!" if creator_turn else "Opponent's turn..."
    creator_text = (
        f"🎮 <b>{turn_text}</b>\n\n"
        f"You: {creator_symbol} | Opponent: {joiner_symbol}\n"
        f"Game code: <code>{game_code}</code>\n\n"
        f"<pre>{_format_board_text(game)}</pre>"
    )

    try:
        await context.bot.send_message(
            chat_id=creator_id,
            text=creator_text,
            reply_markup=reply_markup if creator_turn else None,
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error(f"Failed to send board to creator {creator_id}: {e}")

    # Send to joiner
    if joiner_id:
        joiner_turn = current_turn == joiner_id
        turn_text_joiner = "Your turn!" if joiner_turn else "Opponent's turn..."
        joiner_text = (
            f"🎮 <b>{turn_text_joiner}</b>\n\n"
            f"You: {joiner_symbol} | Opponent: {creator_symbol}\n"
            f"Game code: <code>{game_code}</code>\n\n"
            f"<pre>{_format_board_text(game)}</pre>"
        )

        try:
            await context.bot.send_message(
                chat_id=joiner_id,
                text=joiner_text,
                reply_markup=reply_markup if joiner_turn else None,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error(f"Failed to send board to joiner {joiner_id}: {e}")


async def _handle_multiplayer_game_over(
    context: ContextTypes.DEFAULT_TYPE, game_code: str, winner: Optional[str]
) -> None:
    """
    Handle game over for multiplayer games.

    Args:
        context: The context object for the handler
        game_code: The game code
        winner: The winning symbol or None for draw
    """
    game_data = game_manager.get_game_by_code(game_code)
    if not game_data:
        return

    game = game_data["game"]
    creator_id = game_data["creator_id"]
    joiner_id = game_data["joiner_id"]
    creator_symbol = game_data["creator_symbol"]
    joiner_symbol = game_data["joiner_symbol"]

    board_text = f"<pre>{_format_board_text(game)}</pre>"

    # Update stats for both players
    if winner:
        if winner == creator_symbol:
            game_manager.update_stats(creator_id, "win")
            game_manager.update_stats(joiner_id, "loss")
        else:
            game_manager.update_stats(creator_id, "loss")
            game_manager.update_stats(joiner_id, "win")
    else:
        game_manager.update_stats(creator_id, "draw")
        game_manager.update_stats(joiner_id, "draw")

    # Prepare rematch keyboard
    keyboard = [
        [InlineKeyboardButton("🔄 New Game", callback_data="mode_multi")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send to creator
    creator_stats = game_manager.get_stats(creator_id)
    if winner:
        if winner == creator_symbol:
            creator_result = "🎉 <b>You Won!</b> 🎉"
        else:
            creator_result = "😔 <b>You Lost!</b>"
    else:
        creator_result = "🤝 <b>It's a Draw!</b>"

    creator_message = (
        f"{creator_result}\n\n"
        f"{board_text}\n\n"
        f"📊 Your Stats:\n"
        f"🏆 Wins: {creator_stats['wins']} | "
        f"😔 Losses: {creator_stats['losses']} | "
        f"🤝 Draws: {creator_stats['draws']}"
    )

    try:
        await context.bot.send_message(
            chat_id=creator_id,
            text=creator_message,
            reply_markup=reply_markup,
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error(f"Failed to send game over to creator {creator_id}: {e}")

    # Send to joiner
    if joiner_id:
        joiner_stats = game_manager.get_stats(joiner_id)
        if winner:
            if winner == joiner_symbol:
                joiner_result = "🎉 <b>You Won!</b> 🎉"
            else:
                joiner_result = "😔 <b>You Lost!</b>"
        else:
            joiner_result = "🤝 <b>It's a Draw!</b>"

        joiner_message = (
            f"{joiner_result}\n\n"
            f"{board_text}\n\n"
            f"📊 Your Stats:\n"
            f"🏆 Wins: {joiner_stats['wins']} | "
            f"😔 Losses: {joiner_stats['losses']} | "
            f"🤝 Draws: {joiner_stats['draws']}"
        )

        try:
            await context.bot.send_message(
                chat_id=joiner_id,
                text=joiner_message,
                reply_markup=reply_markup,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error(f"Failed to send game over to joiner {joiner_id}: {e}")

    # Clean up game
    game_manager.delete_multiplayer_game(game_code)
    logger.info(f"Multiplayer game {game_code} ended. Winner: {winner}")


def _format_board_text(game) -> str:
    """
    Format the game board as text.

    Args:
        game: The TicTacToeGame instance

    Returns:
        str: Formatted board string
    """
    board = game.get_board()
    lines = []
    for i, row in enumerate(board):
        formatted_row = " | ".join(cell if cell != " " else " " for cell in row)
        lines.append(f" {formatted_row}")
        if i < 2:
            lines.append("-----------")
    return "\n".join(lines)


async def _send_game_board(message, user_id: int) -> None:
    """
    Send or update the game board with inline keyboard.

    Args:
        message: The message object to edit
        user_id: The user's Telegram ID
    """
    game_data = game_manager.get_game(user_id)
    if not game_data:
        return

    game = game_data["game"]
    board = game.get_board()

    # Create inline keyboard for the board
    keyboard = []
    for row_idx in range(3):
        row_buttons = []
        for col_idx in range(3):
            position = row_idx * 3 + col_idx
            cell = board[row_idx][col_idx]

            # Show X, O, or position number
            if cell == "X":
                button_text = "❌"
            elif cell == "O":
                button_text = "⭕"
            else:
                button_text = str(position + 1)

            row_buttons.append(
                InlineKeyboardButton(
                    button_text,
                    callback_data=f"pos_{position}",
                )
            )
        keyboard.append(row_buttons)

    reply_markup = InlineKeyboardMarkup(keyboard)

    board_text = (
        f"🎮 <b>Your turn!</b>\n\n"
        f"You: {game_data['player_symbol']} | AI: {game_data['ai_symbol']}\n\n"
        f"<pre>{_format_board_text(game)}</pre>"
    )

    try:
        await message.edit_text(
            board_text,
            reply_markup=reply_markup,
            parse_mode="HTML",
        )
    except Exception:
        # If edit fails, send new message
        await message.reply_html(board_text, reply_markup=reply_markup)


async def _send_game_over_message(
    message, game, user_id: int, winner: Optional[str]
) -> None:
    """
    Send game over message with result.

    Args:
        message: The message object
        game: The TicTacToeGame instance
        user_id: The user's Telegram ID
        winner: The winning symbol or None for draw
    """
    game_data = game_manager.get_game(user_id)
    player_symbol = game_data["player_symbol"] if game_data else "X"

    if winner:
        if winner == player_symbol:
            result_text = "🎉 <b>You Won!</b> 🎉"
        else:
            result_text = "😔 <b>AI Won!</b>"
    else:
        result_text = "🤝 <b>It's a Draw!</b>"

    stats = game_manager.get_stats(user_id)
    board_text = f"<pre>{_format_board_text(game)}</pre>"

    final_message = (
        f"{result_text}\n\n"
        f"{board_text}\n\n"
        f"📊 Your Stats:\n"
        f"🏆 Wins: {stats['wins']} | "
        f"😔 Losses: {stats['losses']} | "
        f"🤝 Draws: {stats['draws']}\n\n"
        f"Use /newgame to play again!"
    )

    try:
        await message.edit_text(final_message, parse_mode="HTML")
    except Exception:
        await message.reply_html(final_message)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global error handler for uncaught exceptions.

    Logs errors with full context and notifies the user if possible.
    Handles different types of errors appropriately.

    Args:
        update: The update that caused the error.
        context: The context object containing error information.
    """
    error = context.error

    # Log error with full context
    logger.error(
        f"Exception while handling an update: {type(error).__name__}: {error}",
        exc_info=error,
    )

    # Log update details for debugging
    if isinstance(update, Update):
        logger.error(f"Update that caused error: {update}")

    # Try to notify the user about the error
    try:
        if isinstance(update, Update) and update.effective_message:
            # Determine appropriate error message
            error_message = ErrorMessages.GENERIC_ERROR

            if isinstance(error, TelegramError):
                if "timeout" in str(error).lower():
                    error_message = ErrorMessages.TIMEOUT_ERROR
                elif "network" in str(error).lower():
                    error_message = ErrorMessages.NETWORK_ERROR

            await update.effective_message.reply_text(error_message)
    except Exception as e:
        # If we can't notify the user, just log it
        logger.error(f"Failed to send error message to user: {e}")


def load_environment() -> Optional[str]:
    """
    Load environment variables and retrieve the bot token.

    Returns:
        The bot token if found, None otherwise.

    Raises:
        ValueError: If TELEGRAM_BOT_TOKEN is not set (logged but not raised)
    """
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        logger.critical(
            "TELEGRAM_BOT_TOKEN not found in environment variables. "
            "Please set it in your .env file or environment."
        )
        return None

    logger.info("Environment variables loaded successfully")
    return token


def main() -> None:
    """
    Main function to initialize and run the bot.

    Sets up the application, registers handlers, and starts polling.
    Includes error handling for initialization failures.

    Raises:
        SystemExit: If bot cannot be initialized
    """
    try:
        # Load bot token from environment
        token = load_environment()
        if not token:
            logger.critical("Cannot start bot without TELEGRAM_BOT_TOKEN")
            raise SystemExit(1)

        logger.info("Initializing bot application...")

        # Create the Application with timeout settings
        application = (
            Application.builder()
            .token(token)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .build()
        )
    except Exception as e:
        logger.critical(f"Failed to initialize bot application: {e}", exc_info=True)
        raise SystemExit(1)

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("newgame", newgame_command))
    application.add_handler(CommandHandler("stats", stats_command))

    # Register callback query handlers
    application.add_handler(
        CallbackQueryHandler(mode_selection_callback, pattern="^mode_")
    )
    application.add_handler(
        CallbackQueryHandler(symbol_selection_callback, pattern="^symbol_")
    )
    application.add_handler(
        CallbackQueryHandler(board_button_callback, pattern="^pos_")
    )
    application.add_handler(
        CallbackQueryHandler(multi_create_callback, pattern="^multi_create$")
    )
    application.add_handler(
        CallbackQueryHandler(multi_join_callback, pattern="^multi_join$")
    )
    application.add_handler(
        CallbackQueryHandler(multi_symbol_callback, pattern="^multi_symbol_")
    )
    application.add_handler(
        CallbackQueryHandler(cancel_game_callback, pattern="^cancel_")
    )
    application.add_handler(
        CallbackQueryHandler(multiplayer_board_callback, pattern="^multi_pos_")
    )

    # Register message handler for game codes
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, game_code_message_handler)
    )

    # Register error handler
    application.add_error_handler(error_handler)

    try:
        # Start the bot
        logger.info("Bot initialized successfully. Starting polling...")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.critical(f"Fatal error while running bot: {e}", exc_info=True)
        raise SystemExit(1)
    finally:
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
