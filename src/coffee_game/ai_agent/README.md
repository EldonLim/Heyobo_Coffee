# ☕ Coffee Shop AI Storefront Assistant

A voice-based AI assistant with computer vision that acts as a cute robot mascot for a coffee shop storefront. It detects customers, analyzes their appearance/mood, and generates personalized greetings to engage them in fun coffee games for rewards!

## ✨ Features

- **👤 Person Detection** - Uses webcam + MediaPipe/DeepFace to detect customer traits (emotion, age, gender, clothing)
- **🎯 Personalized Greetings** - AI generates custom opening lines based on detected traits
- **🎙️ Voice Interaction** - Full voice conversation with speech-to-text and text-to-speech
- **🎮 Coffee Games** - Plays quick games with customers for discount rewards
- **🤖 Cute Mascot Persona** - Energetic, coffee-obsessed robot personality

## 🎬 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Customer Approaches                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Person Analyzer 📷                             │
│            (OpenCV + MediaPipe + DeepFace)                       │
│    Detects: emotion, gender, age, shirt color, glasses           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Personalized Greeting Generator 🎯                  │
│                      GPT-4o                                      │
│  "Hey blue shirt! Looking happy — want a coffee deal?"           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Text-to-Speech (TTS) 🔊                         │
│              OpenAI TTS API (shimmer voice)                      │
│                Cute robot mascot voice                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Audio Recorder 🎙️                               │
│                (sounddevice + soundfile)                         │
│              Records customer's response                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Speech-to-Text (STT) 📝                           │
│            OpenAI Whisper API (English)                          │
│            + Hallucination filtering                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Processing 🤖                              │
│                      GPT-4o                                      │
│        Coffee shop persona with conversation history             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        (Loop back to TTS)
```

## 📁 Project Structure

```
python_conversation_ai_agent/
├── main.py                 # Main orchestrator with person detection
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── .env                   # Your API keys (create this)
├── README.md              # This file
└── src/
    ├── __init__.py
    ├── audio_recorder.py  # Records audio from microphone
    ├── speech_to_text.py  # Transcribes audio using Whisper
    ├── llm.py             # GPT-4o with coffee shop persona
    ├── text_to_speech.py  # Converts text to speech (shimmer voice)
    └── person_analyzer.py # Webcam person trait detection
```

## 📦 Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `person_analyzer.py` | Detects person traits via webcam (emotion, gender, age, clothing) |
| `audio_recorder.py` | Records audio from microphone, saves as WAV file |
| `speech_to_text.py` | Transcribes audio to text using OpenAI Whisper API (English, with hallucination filtering) |
| `llm.py` | GPT-4o with coffee shop persona, generates personalized greetings, maintains conversation history |
| `text_to_speech.py` | Converts text to speech using OpenAI TTS (shimmer voice), plays audio |
| `main.py` | Orchestrates person detection → greeting → conversation flow |

## 🚀 Quick Start

### 1. Clone and navigate to the project

```bash
cd python_conversation_ai_agent
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
- `OPENAI_API_KEY` - Get from [OpenAI Platform](https://platform.openai.com/)

### 5. Run the assistant

**Single interaction (with camera):**
```bash
python main.py
```

**Single interaction (without camera):**
```bash
python main.py --no-camera
```

**Continuous mode for multiple customers:**
```bash
python main.py --loop
```

**Continuous mode without camera:**
```bash
python main.py --loop --no-camera
```

## 🎯 Usage Examples

### With Camera (Personalized Greeting)
```
📷 ANALYZING CUSTOMER...
Looking at you for 3 seconds...
✅ Person analyzed!
👤 Detected person: gender: male, age group: 25-35, current mood: happy, wearing a blue shirt

🎯 ENGAGING CUSTOMER...
🤖 AI: "Hey, Mr. Blue Shirt! You're looking chipper — want a coffee deal?"
```

### Without Camera (Generic Greeting)
```
🎯 ENGAGING CUSTOMER...
🤖 AI: "HEY coffee lover! Yeah YOU! I can smell your caffeine craving from here!"
```

## 🔧 Configuration

The assistant is configured with sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Recording duration | 5 seconds | How long to record audio |
| Sample rate | 16000 Hz | Audio sample rate |
| LLM model | gpt-4o | OpenAI model for conversation |
| TTS voice | shimmer | Cute female robot voice |
| Person analysis | 3 seconds | How long to analyze customer |
| STT language | English | Forces English transcription |

## 🎮 Coffee Games & Rewards

The AI plays quick games with customers:
- **Number Game**: "Pick 1, 2, or 3 — one is the PERFECT roast!"
- **Vibe Check**: "Hot or iced? I'll judge your coffee soul!"
- **Secret Ingredient**: "Guess: vanilla, caramel, or hazelnut?"

**Rewards:**
- Win: 10% off your drink
- Lose: 5% off for being a good sport
- Everyone gets something!

## ⚠️ Potential Failure Points & Safeguards

| Issue | Safeguard |
|-------|-----------|
| No speech detected | AI prompts user again with coffee-themed message |
| Whisper hallucination | Filters fake phrases like "Thank you for watching!" |
| Camera not available | Falls back to generic greeting with `--no-camera` |
| Empty transcription | Checks for empty string, prompts user |
| API rate limits | Basic error handling with messages |
| Network issues | Exception handling with user feedback |
| Microphone not found | sounddevice provides clear error |

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4o
- **STT**: OpenAI Whisper API (forced English)
- **TTS**: OpenAI TTS (shimmer voice)
- **Vision**: OpenCV + MediaPipe + DeepFace
- **Audio**: sounddevice + soundfile + pydub

## 🔮 Future Improvements

- Add streaming support for faster responses
- Implement voice activity detection (VAD)
- Support multiple languages
- Add WebSocket support for real-time communication
- Integrate with POS system for automatic discounts
- Add facial recognition for returning customers
- Display promotional visuals on screen

## 📝 License

MIT License - Feel free to use and modify!
