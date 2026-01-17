"""
Text-to-Speech Module
Converts text to speech using OpenAI TTS and plays it locally.
"""

import os
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def text_to_speech(text: str, output_path: str = None) -> str:
    """
    Convert text to speech using OpenAI TTS API and save as MP3.
    
    Args:
        text: The text to convert to speech
        output_path: Optional path to save the audio file
        
    Returns:
        Path to the saved audio file
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("🔈 Generating speech...")
    
    # Generate speech
    response = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=text
    )
    
    # Determine output path
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp3")
    
    # Save the audio file
    response.stream_to_file(output_path)
    
    return output_path


def play_audio(audio_path: str) -> None:
    """
    Play an audio file using the system's default audio player.
    
    Args:
        audio_path: Path to the audio file to play
    """
    import subprocess
    import platform
    
    print("🔊 Playing audio...")
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        subprocess.run(["afplay", audio_path], check=True)
    elif system == "Linux":
        # Try different players
        for player in ["aplay", "paplay", "ffplay"]:
            try:
                subprocess.run([player, audio_path], check=True)
                break
            except FileNotFoundError:
                continue
    elif system == "Windows":
        import winsound
        # For MP3, we need to convert or use a different approach
        subprocess.run(["powershell", "-c", f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'], check=True)
    else:
        print(f"⚠️ Unsupported platform: {system}")


def speak(text: str) -> None:
    """
    Convert text to speech and play it immediately.
    
    Args:
        text: The text to speak
    """
    audio_path = text_to_speech(text)
    play_audio(audio_path)
    
    # Clean up temp file
    try:
        os.remove(audio_path)
    except:
        pass


if __name__ == "__main__":
    # Test TTS
    import sys
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = "Hello! I am your AI assistant. How can I help you today?"
    
    speak(test_text)
