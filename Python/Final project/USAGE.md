# Usage Guide - Tic-Tac-Toe Telegram Bot

Complete guide on how to use the Tic-Tac-Toe Telegram Bot.

## Table of Contents

- [Getting Started](#getting-started)
- [Commands Reference](#commands-reference)
- [Single-Player Mode](#single-player-mode)
- [Multiplayer Mode](#multiplayer-mode)
- [Statistics](#statistics)

## Getting Started

### Finding Your Bot

1. Open Telegram
2. Search for your bot by username (e.g., `@my_tictactoe_bot`)
3. Click on the bot to open the chat
4. Click "Start" or send `/start`

### First Interaction

When you start the bot, you'll see a welcome message with available commands:

```
👋 Hello @username!

Welcome to Tic-Tac-Toe Bot! 🎮

Available commands:
/start - Start the bot
/help - Show help information
/newgame - Start a new game
/stats - View your statistics
```

## Commands Reference

### `/start`
**Purpose:** Initialize the bot and see welcome message

**Usage:**
```
/start
```

**Response:** Welcome message with command list

---

### `/help`
**Purpose:** Display help information and game rules

**Usage:**
```
/help
```

**Response:** Detailed help message with:
- Command descriptions
- How to play instructions
- Game rules

---

### `/newgame`
**Purpose:** Start a new Tic-Tac-Toe game

**Usage:**
```
/newgame
```

**Response:** Game mode selection menu with options:
- 🤖 Single Player (vs AI)
- 👥 Multiplayer

**Note:** If you have an active game, it will be replaced with a new one.

---

### `/stats`
**Purpose:** View your game statistics

**Usage:**
```
/stats
```

**Response:** Your statistics including:
- 🎮 Total Games
- 🏆 Wins
- 😔 Losses
- 🤝 Draws
- 📈 Win Rate

**Example:**
```
📊 Your Statistics

🎮 Total Games: 15
🏆 Wins: 8
😔 Losses: 5
🤝 Draws: 2
📈 Win Rate: 53.3%
```

## Single-Player Mode

Play against an AI opponent.

### Starting a Single-Player Game

1. **Send `/newgame`**
   ```
   /newgame
   ```

2. **Select Single Player**
   - Click "🤖 Single Player (vs AI)" button

3. **Choose Your Symbol**
   - Click "❌ Play as X (go first)" to play first
   - Click "⭕ Play as O (go second)" to let AI go first

4. **Game Starts**
   - If you chose X, you make the first move
   - If you chose O, AI makes the first move

### Making Moves

The game board appears as a 3x3 grid of buttons:

```
 1 | 2 | 3
-----------
 4 | 5 | 6
-----------
 7 | 8 | 9
```

**To make a move:**
1. Click on any numbered button (1-9)
2. Your symbol (X or O) will appear in that position
3. AI will automatically make its move
4. Continue until someone wins or it's a draw

### Game Board Display

During the game, you'll see:
```
🎮 Your turn!

You: X | AI: O

 X |   | O
-----------
   | X |  
-----------
 O |   |  
```

### Winning

When you win:
```
🎉 You Won! 🎉

 X | O | X
-----------
 O | X | O
-----------
 X |   |  

📊 Your Stats:
🏆 Wins: 9 | 😔 Losses: 5 | 🤝 Draws: 2

Use /newgame to play again!
```

### Losing

When AI wins:
```
😔 AI Won!

 O | X | O
-----------
 X | O | X
-----------
 O | X |  

📊 Your Stats:
🏆 Wins: 8 | 😔 Losses: 6 | 🤝 Draws: 2

Use /newgame to play again!
```

### Draw

When it's a draw:
```
🤝 It's a Draw!

 X | O | X
-----------
 O | X | O
-----------
 O | X | O

📊 Your Stats:
🏆 Wins: 8 | 😔 Losses: 5 | 🤝 Draws: 3

Use /newgame to play again!
```

## Multiplayer Mode

Play with friends using unique game codes.

### Creating a Multiplayer Game

1. **Send `/newgame`**
   ```
   /newgame
   ```

2. **Select Multiplayer**
   - Click "👥 Multiplayer" button

3. **Create Game**
   - Click "🎮 Create Game" button

4. **Choose Your Symbol**
   - Click "❌ Play as X (go first)"
   - Click "⭕ Play as O (go second)"

5. **Share Game Code**
   - You'll receive a 6-character game code
   - Share this code with your friend

**Example:**
```
🎮 Game Created!

Your symbol: X
Game code: ABC123

⏳ Waiting for opponent to join...

Share this code with your friend!
```

### Joining a Multiplayer Game

1. **Send `/newgame`**
   ```
   /newgame
   ```

2. **Select Multiplayer**
   - Click "👥 Multiplayer" button

3. **Join Game**
   - Click "🔗 Join Game" button

4. **Enter Game Code**
   - Type the 6-character code your friend shared
   - Example: `ABC123`

5. **Game Starts**
   - Both players will be notified
   - Game begins automatically

**Example:**
```
✅ Joined Game!

Your symbol: O
Game code: ABC123

Game is starting...
```

### Playing Multiplayer

**Turn-Based Gameplay:**
- Only the current player can make moves
- Other player sees "Opponent's turn..."
- Board updates automatically for both players

**Your Turn:**
```
🎮 Your turn!

You: X | Opponent: O
Game code: ABC123

 X |   | O
-----------
   |   |  
-----------
   |   |  
```

**Opponent's Turn:**
```
🎮 Opponent's turn...

You: O | Opponent: X
Game code: ABC123

 X |   | O
-----------
   |   |  
-----------
   |   |  
```

### Canceling a Game

If you're waiting for an opponent, you can cancel:
- Click "❌ Cancel Game" button
- Game will be deleted
- You can create a new game

### Game Completion

Both players receive the result:

**Winner:**
```
🎉 You Won! 🎉

 X | O | X
-----------
 O | X | O
-----------
 X |   |  

📊 Your Stats:
🏆 Wins: 10 | 😔 Losses: 5 | 🤝 Draws: 2
```

**Loser:**
```
😔 You Lost!

 X | O | X
-----------
 O | X | O
-----------
 X |   |  

📊 Your Stats:
🏆 Wins: 8 | 😔 Losses: 6 | 🤝 Draws: 2
```


## Statistics

### Tracked Metrics

The bot tracks these statistics for each user:

- **Total Games:** Sum of all games played
- **Wins:** Games you won
- **Losses:** Games you lost
- **Draws:** Games that ended in a tie
- **Win Rate:** Percentage of games won

### Viewing Statistics

Send `/stats` anytime to see your current statistics:

```
📊 Your Statistics

🎮 Total Games: 25
🏆 Wins: 15
😔 Losses: 7
🤝 Draws: 3
📈 Win Rate: 60.0%
```

### Statistics Persistence

- Statistics are saved across sessions
- Stats persist even if you start a new game
- Both single-player and multiplayer games count
