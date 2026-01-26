# Quick Start Guide - Tic-Tac-Toe Telegram Bot

Get your bot running in 5 minutes! ⚡

---

## Prerequisites

- Python 3.11 or higher
- Telegram account
- 5 minutes of your time

---

## Step 1: Get Your Bot Token (2 minutes)

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Follow the prompts to name your bot
4. Copy the bot token (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

---

## Step 2: Setup Project (2 minutes)

```bash
# Clone the repository
git clone https://github.com/...

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Configure Bot (30 seconds)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your bot token
# Replace YOUR_BOT_TOKEN_HERE with your actual token
```

Your `.env` file should look like:
```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
ENVIRONMENT=development
```

---

## Step 4: Run the Bot (30 seconds)

```bash
python3 -m src.bot
```

You should see:
```
2026-01-26 11:30:00 - Bot started successfully!
2026-01-26 11:30:00 - Bot is running. Press Ctrl+C to stop.
```

---

## Step 5: Test Your Bot (1 minute)

1. Open Telegram
2. Search for your bot by username
3. Send `/start` command
4. Click "🎮 New Game" button
5. Choose your symbol (X or O)
6. Start playing!

---

## Quick Test Commands

Try these commands in your bot:

```
/start          - Welcome message and main menu
/newgame        - Start a new game
/stats          - View your statistics
/help           - Get help
/join ABCD12    - Join multiplayer game (use actual code)
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "TELEGRAM_BOT_TOKEN environment variable is not set"
**Solution**: Check your `.env` file exists and contains your bot token:
```bash
cat .env  # Should show your token
```

### Issue: Bot doesn't respond
**Solution**: 
1. Check bot is running (you should see log messages)
2. Verify token is correct in `.env`
3. Make sure you're messaging the correct bot

### Issue: "Connection error"
**Solution**: Check your internet connection and try again

---

## Next Steps

Now that your bot is running:

1. **Play Some Games**: Test both single-player and multiplayer modes
2. **Check Statistics**: Use `/stats` to see your win/loss record
3. **Read Full Documentation**: 
   - [USAGE.md](USAGE.md) - Complete user guide
   - [DEVELOPMENT.md](DEVELOPMENT.md) - For customization
   - [DEPLOYMENT.md](DEPLOYMENT.md) - For production deployment

---

## Quick Reference

### File Structure
```
Final project/
├── src/
│   └── bot.py          # Main bot file (run this)
├── .env                # Your configuration (create from .env.example)
├── requirements.txt    # Dependencies
└── logs/              # Log files (auto-created)
```

### Important Commands
```bash
# Start bot
python -m src.bot

# Run tests
pytest

# Check code quality
black src/ tests/
flake8 src/ tests/
```

### Environment Variables
```bash
TELEGRAM_BOT_TOKEN=your_token_here    # Required
ENVIRONMENT=development               # Optional (development/production)
```

---

## Stopping the Bot

Press `Ctrl+C` in the terminal where the bot is running.

---

## Getting Help

- **Documentation**: Check [README.md](README.md) for overview
- **Setup Issues**: See [SETUP.md](SETUP.md) for detailed setup
- **Usage Questions**: Read [USAGE.md](USAGE.md) for features
- **Development**: See [DEVELOPMENT.md](DEVELOPMENT.md) for code details

---

## Success Checklist

- [ ] Python 3.11+ installed
- [ ] Bot token obtained from @BotFather
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] `.env` file created with bot token
- [ ] Bot running without errors
- [ ] Bot responds to `/start` command
- [ ] Successfully played a game

---

**Congratulations! Your bot is now running! 🎉**

Start playing and enjoy your Tic-Tac-Toe bot!

---

*For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md)*