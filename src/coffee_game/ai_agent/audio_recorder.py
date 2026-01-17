"""
Audio Recorder Module
Records audio from the microphone and saves it as a WAV file.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path


def record_audio(
    duration: float = 5.0,
    sample_rate: int = 16000,
    output_path: str = "recording.wav"
) -> str:
    """
    Record audio from the microphone and save as WAV file.
    
    Args:
        duration: Recording duration in seconds (default: 5.0)
        sample_rate: Audio sample rate in Hz (default: 16000)
        output_path: Path to save the WAV file
        
    Returns:
        Path to the saved audio file
    """
    print(f"🎤 Recording for {duration} seconds...")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    
    # Wait for recording to complete
    sd.wait()
    
    print("✅ Recording complete!")
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as WAV file
    sf.write(output_path, audio_data, sample_rate)
    
    return str(output_file.absolute())


if __name__ == "__main__":
    # Test the recorder
    audio_path = record_audio(duration=3.0, output_path="test_recording.wav")
    print(f"Audio saved to: {audio_path}")
