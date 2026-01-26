# Tic-Tac-Toe Telegram Bot 🎮

A feature-rich Telegram bot that allows users to play Tic-Tac-Toe (Крестики-нолики) with AI or against other players in multiplayer mode.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Features

### Single-Player Mode
- 🤖 Play against AI opponent
- ❌⭕ Choose your symbol (X or O)
- 🎯 Smart AI that makes strategic moves
- 📊 Track your wins, losses, and draws

### Multiplayer Mode
- 👥 Play with friends via unique game codes
- 🔗 Easy game sharing with 6-character codes
- ⏱️ Real-time turn-based gameplay
- 🎮 Interactive inline keyboard controls

### Additional Features
- 📈 Personal statistics tracking
- 🔄 Game state persistence
- 🛡️ Robust error handling
- 📝 Comprehensive logging
- 🌐 Support for concurrent games
- ⚡ Fast and responsive

## 📋 Prerequisites

- Python 3.8 or higher
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- pip (Python package manager)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Final\ project
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your bot token:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
ENVIRONMENT=development
```

### 5. Run the Bot

```bash
python -m src.bot
```

## 📖 Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the bot and see welcome message |
| `/help` | Show help information and game rules |
| `/newgame` | Start a new game (single or multiplayer) |
| `/stats` | View your game statistics |

### Playing Single-Player

1. Send `/newgame` to the bot
2. Select "🤖 Single Player (vs AI)"
3. Choose your symbol (X or O)
4. Click on the board buttons to make your moves
5. Try to get three in a row to win!

### Playing Multiplayer

**Creating a Game:**
1. Send `/newgame` to the bot
2. Select "👥 Multiplayer"
3. Click "🎮 Create Game"
4. Choose your symbol (X or O)
5. Share the 6-character game code with your friend

**Joining a Game:**
1. Send `/newgame` to the bot
2. Select "👥 Multiplayer"
3. Click "🔗 Join Game"
4. Enter the game code provided by your friend
5. Start playing!

## 🏗️ Project Structure

```
Final project/
├── config/
│   └── settings.py          # Configuration settings
├── src/
│   ├── __init__.py
│   ├── bot.py               # Main bot logic and handlers
│   ├── game.py              # Tic-Tac-Toe game logic
│   ├── game_manager.py      # Game session management
│   ├── ai_player.py         # AI opponent implementation
│   ├── error_handlers.py    # Error handling utilities
│   └── logging_config.py    # Logging configuration
├── tests/
│   ├── test_game.py         # Game logic tests
│   ├── test_ai_player.py    # AI player tests
│   ├── test_game_manager.py # Game manager tests
│   ├── test_multiplayer.py  # Multiplayer tests
│   └── test_error_handling.py # Error handling tests
├── logs/                    # Log files (auto-generated)
├── .env                     # Environment variables (create from .env.example)
├── .env.example             # Example environment file
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_game.py
```

### Run with Coverage

```bash
pytest --cov=src tests/
```

### Run with Verbose Output

```bash
pytest -v
```

## 🎨 Code Quality

### Format Code with Black

```bash
black src/ tests/
```

### Check Code Style with Flake8

```bash
flake8 src/ tests/
```

### Sort Imports with isort

```bash
isort src/ tests/
```

### Run All Quality Checks

```bash
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
```

## 📊 Statistics

The bot tracks the following statistics for each user:
- 🏆 Total wins
- 😔 Total losses
- 🤝 Total draws
- 📈 Win rate percentage

View your stats anytime with the `/stats` command.

## 🛡️ Error Handling

The bot includes comprehensive error handling for:
- ❌ Network errors and timeouts
- 🔒 Invalid user input
- 🎮 Game state inconsistencies
- 👥 Concurrent game attempts
- 📡 Telegram API failures
- ⏱️ Rate limiting

All errors are logged with full context for debugging.

## 📝 Logging

Logs are stored in the `logs/` directory:
- `bot.log` - General application logs
- `errors.log` - Error-specific logs

Log files automatically rotate when they reach 10MB, keeping the last 5 backups.

## 🔧 Configuration

Configuration settings can be modified in [`config/settings.py`](config/settings.py):

- Game settings (board size, symbols)
- Bot settings (timeouts, retries)
- Logging configuration
- Error messages

## 🐛 Troubleshooting

### Bot doesn't start

**Problem:** `TELEGRAM_BOT_TOKEN not found`

**Solution:** Make sure you've created a `.env` file with your bot token:
```env
TELEGRAM_BOT_TOKEN=your_token_here
```

### Import errors

**Problem:** `ModuleNotFoundError`

**Solution:** Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Tests fail

**Problem:** Tests fail with import errors

**Solution:** Make sure you're running tests from the project root:
```bash
pytest
```

### Bot is slow or unresponsive

**Problem:** Bot takes long to respond

**Solution:** Check your internet connection and Telegram API status. The bot includes automatic retry logic for transient failures.

## 📚 Additional Documentation

- [SETUP.md](SETUP.md) - Detailed setup instructions
- [USAGE.md](USAGE.md) - Comprehensive usage guide
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and code quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for all public functions
- Format code with `black`
- Sort imports with `isort`
- Check with `flake8`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created as part of the Yandex Advanced Python Course 2023.

## 🙏 Acknowledgments

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) - Telegram Bot API wrapper
- [pytest](https://pytest.org/) - Testing framework
- [black](https://github.com/psf/black) - Code formatter

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [documentation](SETUP.md)
3. Open an issue on GitHub

## 🔮 Future Improvements

- [ ] Add difficulty levels for AI (easy, medium, hard)
- [ ] Implement minimax algorithm for unbeatable AI
- [ ] Add game history and replay functionality
- [ ] Support for larger board sizes (4x4, 5x5)
- [ ] Leaderboard system
- [ ] Tournament mode
- [ ] Custom themes and emojis
- [ ] Multi-language support

---

Made with ❤️ using Python and python-telegram-bot
