# ☕ Coffee Catcher Game

A fun Python game where you catch falling coffee beans to fill your cup while avoiding bombs! Features an AI assistant with animated character (HeyoBo), hand gesture controls, and discount voucher rewards.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.6+-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)

## ✨ Features

- 🎮 **Classic arcade gameplay** - Catch coffee beans, avoid bombs
- 🖐️ **Hand gesture control** - Play using webcam and hand movements
- 🤖 **AI Assistant** - Pre-game conversation with personalized greetings
- 👀 **Person detection** - Detects emotions, age, gender using DeepFace
- �️ **Gaze detection** - Requires user attention before starting conversation
- �🐱 **HeyoBo Character** - Animated assistant with speaking/listening states
- 🎟️ **Voucher Rewards** - Win discount QR codes based on performance

## 🎮 Gameplay

- **Objective**: Fill your coffee cup before time runs out
- **Controls**: 
  - `←` / `→` Arrow keys to move the cup (keyboard mode)
  - Hand gestures via webcam (hand control mode)
  - `R` to restart after game over
  - `ESC` to quit

## 🎯 Rules

- ☕ **Coffee beans**: +10% fullness
- 💣 **Bombs**: Instant game over!
- ⏱️ **Timer**: 30 seconds to fill your cup

## 🎟️ Voucher Rewards

| Cup Fullness | Discount |
|--------------|----------|
| 0-29%        | 5% OFF   |
| 30-59%       | 10% OFF  |
| 60-89%       | 15% OFF  |
| 90-100%      | 20% OFF  |

## 📦 Installation

### Requirements
- Python 3.12 (required for DeepFace/TensorFlow compatibility)
- macOS: Tkinter (`brew install python-tk@3.12`)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd python_coffee_game
   ```

2. **Create virtual environment with Python 3.12**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

   > **Note for macOS**: If pygame installation fails, install SDL2 first:
   > ```bash
   > brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
   > ```

4. **Set up OpenAI API key** (for AI assistant)
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

## 🚀 Running the Game

### Full Experience (Recommended)
```bash
python -m coffee_game.main --heyobo --hand-control --gaze
```

### All Command-Line Options

| Flag | Short | Description |
|------|-------|-------------|
| `--hand-control` | `-hc` | Enable hand gesture control via webcam |
| `--heyobo` | | Show HeyoBo animated character during AI conversation |
| `--gaze` | `-g` | Require user to look at camera for 2 seconds before starting |
| `--no-ai` | | Skip AI assistant, start game directly |
| `--no-camera` | | Disable camera for person detection (uses generic greeting) |

### Examples

```bash
# Full experience with gaze detection
python -m coffee_game.main --heyobo --hand-control --gaze

# AI assistant with HeyoBo + hand control (no gaze check)
python -m coffee_game.main --heyobo --hand-control

# AI assistant with HeyoBo, no camera detection
python -m coffee_game.main --heyobo --no-camera

# Hand control only, skip AI
python -m coffee_game.main --no-ai --hand-control

# Keyboard mode only
python -m coffee_game.main --no-ai
```

### 🖐️ Hand Gesture Controls

When using `--hand-control`, an instructions screen will appear:

- ☝️ **Odd finger count (1, 3, 5)** → Move RIGHT
- ✌️ **Even finger count (2, 4)** → Move LEFT
- ✊ **Fist (0 fingers)** → START GAME

## 🤖 AI Assistant

The AI assistant (powered by OpenAI) greets the user before the game starts. With camera enabled, it uses DeepFace to detect:
- 😊 Emotion (happy, sad, neutral, etc.)
- 👤 Age estimation
- 🚻 Gender
- 👓 Glasses detection

The assistant personalizes greetings based on these observations.

## �️ Gaze Detection

When using `--gaze`, the system requires user attention before starting:
- Uses MediaPipe Face Landmarker to track eye iris positions
- User must look directly at the camera for 2 seconds
- Shows a live preview with progress bar
- Ensures user engagement before AI conversation begins

This is useful for kiosk/booth deployments where you want to ensure the user is actively engaged.

## �🐱 HeyoBo Character

HeyoBo is an animated character that appears during the AI conversation:
- **Idle state** - Default pose
- **Listening state** - When waiting for user input
- **Speaking state** - When AI is talking

HeyoBo uses GIF animations for smooth transitions between states.

## 🔧 Integration

The game is designed as a class for easy integration into other projects:

```python
from coffee_game.game import CoffeeGame

# Basic usage - runs the complete game
game = CoffeeGame()
result = game.run()
print(f"Won: {result['win']}, Fullness: {result['fullness']}%")
game.quit()
```

### Custom Configuration

```python
game = CoffeeGame(
    width=800,              # Screen width
    height=600,             # Screen height
    game_time=60,           # Time limit in seconds
    spawn_interval=500,     # Milliseconds between spawns
    bomb_chance=0.15,       # 15% chance for bombs
    use_hand_control=True,  # Enable hand gesture control
    screen=my_screen        # Use existing pygame screen
)
```

### Frame-by-Frame Control

For integration into existing game loops:

```python
game = CoffeeGame(screen=existing_screen)
game.reset()

# In your main loop
while game.running:
    for event in pygame.event.get():
        game.handle_event(event)
    
    game.update()
    game.draw()
    pygame.display.flip()

# Access game state
print(game.fullness, game.win, game.game_over, game.remaining)
```

## 📁 Project Structure

```
python_coffee_game/
├── src/
│   └── coffee_game/
│       ├── __init__.py
│       ├── main.py              # Entry point with CLI
│       ├── game.py              # CoffeeGame class
│       ├── hand_control.py      # Hand gesture detection
│       ├── heyobo.py            # HeyoBo animated character
│       ├── gaze_detector.py     # Eye gaze detection
│       ├── hand_landmarker.task # MediaPipe model
│       ├── ai_agent/
│       │   ├── init.py          # AI conversation orchestration
│       │   ├── llm.py           # OpenAI LLM integration
│       │   ├── person_analyzer.py  # DeepFace analysis
│       │   ├── speech_to_text.py   # Whisper STT
│       │   ├── text_to_speech.py   # OpenAI TTS
│       │   └── audio_recorder.py   # Microphone input
│       └── assets/
│           ├── bean.png         # Coffee bean sprite
│           ├── bomb.png         # Bomb sprite
│           ├── 0-100%.png       # Cup fullness images
│           ├── *%OFF.png        # QR code vouchers
│           ├── states/          # HeyoBo static states
│           └── transitions/     # HeyoBo GIF animations
├── pyproject.toml
├── requirements.txt
├── .env                         # OpenAI API key
└── README.md
```

## 🎨 Customization

The `CoffeeGame` class exposes these properties you can modify:

| Property | Description |
|----------|-------------|
| `cup_speed` | How fast the cup moves (default: 7) |
| `cup_width` | Cup width in pixels (default: 100) |
| `cup_height` | Cup height in pixels (default: 100) |

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for AI assistant (LLM, STT, TTS) |

## 📄 License

MIT License
