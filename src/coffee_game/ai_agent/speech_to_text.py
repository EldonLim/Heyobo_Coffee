"""
Speech-to-Text Module
Transcribes audio to text using OpenAI Whisper API.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Common Whisper hallucinations when there's silence (exact matches only)
HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thank you for watching!",
    "thanks for watching",
    "thanks for watching!",
    "thank you for listening",
    "thank you for listening!",
    "thanks for listening",
    "thanks for listening!",
    "subscribe",
    "like and subscribe",
    "see you next time",
    "see you next time!",
    "bye bye",
    "bye bye!",
    "bye",
    "goodbye",
    "goodbye!",
    "the end",
    "the end.",
    "music",
    "[music]",
    "applause",
    "[applause]",
    "silence",
    "[silence]",
    "...",
    "you",
    "."
}


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file to text using OpenAI Whisper API.
    
    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        
    Returns:
        Transcribed text as a string
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("🔊 Transcribing audio...")
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
            response_format="text"
        )
    
    # Handle both string and object response formats
    text = transcription if isinstance(transcription, str) else transcription.text
    text = text.strip()
    
    # Check for hallucinations (exact match only for common silence phrases)
    if text.lower() in HALLUCINATION_PHRASES or text == "":
        print("📝 Transcription: [silence detected - hallucination filtered]")
        return ""
    
    print(f"📝 Transcription: {text}")
    
    return text


if __name__ == "__main__":
    # Test with an existing audio file
    import sys
    if len(sys.argv) > 1:
        result = transcribe_audio(sys.argv[1])
        print(f"Result: {result}")
    else:
        print("Usage: python speech_to_text.py <audio_file_path>")
