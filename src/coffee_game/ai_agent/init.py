#!/usr/bin/env python3
"""
Storefront AI Assistant with Person Detection
A voice-based AI assistant that:
1. Detects person traits (emotion, age, gender, clothing)
2. Generates personalized greeting based on traits
3. Records their voice response
4. Plays a quick game
5. Offers rewards to encourage shop entry

Usage:
    python main.py              # Single interaction
    python main.py --loop       # Continuous mode for multiple customers
    python main.py --no-camera  # Run without camera (uses generic greeting)
"""

import os
import sys
import tempfile
import cv2
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.coffee_game.ai_agent.audio_recorder import record_audio
from src.coffee_game.ai_agent.speech_to_text import transcribe_audio
from src.coffee_game.ai_agent.llm import get_llm_response, get_opening_line, get_silence_response, reset_conversation, get_personalized_opening, should_start_game
from src.coffee_game.ai_agent.text_to_speech import speak
from src.coffee_game.ai_agent.person_analyzer import PersonAnalyzer

# Global HeyoBo reference for UI updates
_heyobo_instance = None


def set_heyobo(heyobo):
    """Set the HeyoBo instance for UI updates."""
    global _heyobo_instance
    _heyobo_instance = heyobo


def get_heyobo():
    """Get the current HeyoBo instance."""
    return _heyobo_instance


def _update_heyobo_speak(text: str):
    """Update HeyoBo to speaking state with text."""
    if _heyobo_instance:
        _heyobo_instance.speak(text, typewriter=True, speed=30)
        # Process Tkinter events to show the animation
        _heyobo_instance.root.update()


def _update_heyobo_listen(text: str = "Listening..."):
    """Update HeyoBo to listening state."""
    if _heyobo_instance:
        _heyobo_instance.listen(text)
        _heyobo_instance.root.update()


def _update_heyobo_idle():
    """Update HeyoBo to idle state."""
    if _heyobo_instance:
        _heyobo_instance.idle()
        _heyobo_instance.root.update()


def validate_environment() -> bool:
    """Check that required API keys are set."""
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("❌ Missing required environment variables:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n📝 Please copy .env.example to .env and add your API keys.")
        return False
    return True


# Global person analyzer instance
person_analyzer = None


def init_person_analyzer():
    """Initialize the person analyzer (lazy loading)."""
    global person_analyzer
    if person_analyzer is None:
        print("📷 Initializing camera and person detection...")
        person_analyzer = PersonAnalyzer(analyze_interval=5, output_interval=30)
    return person_analyzer


def analyze_person(duration_seconds: float = 3.0) -> dict:
    """
    Capture and analyze a person using the webcam.
    
    Args:
        duration_seconds: How long to analyze the person
        
    Returns:
        Dict with person traits (emotion, gender, age, shirt_color, wearing_glasses)
        Each field contains a list of all detected values during the analysis period.
        The most common value can be used as the final result.
    """
    analyzer = init_person_analyzer()
    
    print("\n" + "="*50)
    print("📷 ANALYZING CUSTOMER...")
    print("="*50)
    print(f"Looking at you for {duration_seconds} seconds...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Could not open camera, using generic greeting")
        return {
            "wearing_glasses": [],
            "gender": [],
            "shirt_color": [],
            "age": [],
            "emotion": []
        }
    
    # Collect all readings during the analysis period
    import time
    from collections import Counter
    
    start_time = time.time()
    all_readings = {
        "wearing_glasses": [],
        "gender": [],
        "shirt_color": [],
        "age": [],
        "emotion": []
    }
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        result = analyzer.analyze_frame(frame)
        
        # Show preview window only if HeyoBo is not active (avoid GUI conflicts)
        if _heyobo_instance is None:
            annotated = analyzer.draw_annotations(frame.copy(), result)
            cv2.imshow("Customer Detection", annotated)
            cv2.waitKey(1)
        
        # Collect readings from detected persons
        if result.get("persons"):
            for person in result["persons"]:
                if person.get("wearing_glasses") is not None:
                    all_readings["wearing_glasses"].append(person["wearing_glasses"])
                if person.get("gender") and person["gender"] != "unknown":
                    all_readings["gender"].append(person["gender"])
                if person.get("shirt_color") and person["shirt_color"] != "unknown":
                    all_readings["shirt_color"].append(person["shirt_color"])
                if person.get("age") and person["age"] != "unknown":
                    all_readings["age"].append(person["age"])
                if person.get("emotion"):
                    all_readings["emotion"].append(person["emotion"])
    
    cap.release()
    if _heyobo_instance is None:
        cv2.destroyAllWindows()
    
    # Determine most common values for final result
    final_traits = {}
    
    for key, values in all_readings.items():
        if values:
            if key == "wearing_glasses":
                # For boolean, use majority vote
                final_traits[key] = Counter(values).most_common(1)[0][0]
            else:
                # For strings, use most common value
                final_traits[key] = Counter(values).most_common(1)[0][0]
    
    if final_traits:
        print("✅ Person analyzed!")
        print(f"   Detected: {final_traits}")
    else:
        print("⚠️ Could not detect person clearly, using generic greeting")
    
    # Return both raw readings and final determined values
    return {
        "wearing_glasses": all_readings["wearing_glasses"],
        "gender": all_readings["gender"],
        "shirt_color": all_readings["shirt_color"],
        "age": all_readings["age"],
        "emotion": all_readings["emotion"],
        "final": final_traits  # Most common values for easy access
    }


def run_conversation_turn(duration: float = 5.0, is_first_turn: bool = False, person_traits: dict = None) -> tuple[bool, bool]:
    """
    Run a single conversation turn.
    
    Args:
        duration: Recording duration in seconds
        is_first_turn: If True, AI speaks first with an opening line
        person_traits: Dict of person traits for personalized greeting (only used on first turn)
        
    Returns:
        Tuple of (success, should_start_game) - success is True if turn completed,
        should_start_game is True if user agreed to play
    """
    try:
        # If first turn, AI speaks first to hook the customer
        if is_first_turn:
            print("\n" + "="*50)
            print("🎯 ENGAGING CUSTOMER...")
            print("="*50)
            
            # Use personalized opening if we have person traits
            # Extract final aggregated values if available
            if person_traits:
                final_traits = person_traits.get("final", person_traits)
                opening = get_personalized_opening(final_traits)
            else:
                opening = get_opening_line()
            
            print(f"\n🤖 AI: \"{opening}\"")
            _update_heyobo_speak(opening)  # Update HeyoBo to speaking state
            speak(opening)
        
        # Step 1: Record audio
        print("\n" + "="*50)
        print("🎙️  LISTENING...")
        print("="*50)
        
        _update_heyobo_listen("Listening...")  # Update HeyoBo to listening state
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        
        record_audio(duration=duration, output_path=audio_path)
        
        # Step 2: Transcribe audio to text
        print("\n" + "="*50)
        print("📝 TRANSCRIBING...")
        print("="*50)
        
        user_text = transcribe_audio(audio_path)
        
        # Clean up audio file
        try:
            os.remove(audio_path)
        except:
            pass
        
        # Check if transcription is empty
        if not user_text or user_text.strip() == "":
            print("⚠️  No speech detected - prompting user again")
            
            # Get a silence response and speak it
            silence_response = get_silence_response()
            print(f"\n🤖 AI: \"{silence_response}\"")
            _update_heyobo_speak(silence_response)  # Update HeyoBo to speaking state
            speak(silence_response)
            
            return True, False  # Return True to continue the conversation, False for no game
        
        print(f"\n👤 You said: \"{user_text}\"")
        
        # Step 3: Get GPT-4's response
        print("\n" + "="*50)
        print("🤖 THINKING...")
        print("="*50)
        
        response = get_llm_response(user_text)
        
        # Check if game should start
        start_game, clean_response = should_start_game(response)
        
        print(f"\n🤖 AI: \"{clean_response}\"")
        
        # Step 4: Convert response to speech and play
        print("\n" + "="*50)
        print("🔊 SPEAKING...")
        print("="*50)
        
        _update_heyobo_speak(clean_response)  # Update HeyoBo to speaking state
        speak(clean_response)
        
        _update_heyobo_idle()  # Return to idle after speaking
        
        print("\n✅ Turn complete!")
        
        if start_game:
            print("\n🎮 USER AGREED TO PLAY! Starting game...")
        
        return True, start_game
        
    except Exception as e:
        print(f"❌ Error during conversation turn: {e}")
        return False, False


def run_ai_assistant(use_camera: bool = True, max_turns: int = 6) -> bool:
    """
    Run the AI assistant until user agrees to play or conversation ends.
    
    Args:
        use_camera: Whether to use camera for person detection
        max_turns: Maximum number of conversation turns
        
    Returns:
        True if user agreed to play the game, False otherwise
    """
    print("\n" + "="*50)
    print("🏪 COFFEE SHOP AI ASSISTANT")
    print("="*50)
    
    # Reset conversation for fresh start
    reset_conversation()
    
    # Analyze person if camera is enabled
    person_traits = {}
    if use_camera:
        person_traits = analyze_person(duration_seconds=3.0)
    
    # First turn: AI initiates with personalized greeting
    success, start_game = run_conversation_turn(duration=5.0, is_first_turn=True, person_traits=person_traits)
    
    if start_game:
        return True
    
    # Continue conversation until user agrees or max turns reached
    for _ in range(max_turns - 1):
        success, start_game = run_conversation_turn(duration=5.0, is_first_turn=False)
        if start_game:
            return True
        if not success:
            break
    
    print("\n👋 Conversation ended without starting game.")
    return False


def run_conversation_loop(use_camera: bool = True):
    """Run continuous conversation mode for multiple customers."""
    print("\n" + "="*50)
    print("🏪 STOREFRONT AI ASSISTANT")
    print("="*50)
    print("Press Ctrl+C to exit")
    if use_camera:
        print("📷 Camera mode: ON - Will analyze customers")
    else:
        print("📷 Camera mode: OFF - Using generic greetings")
    print("="*50)
    
    while True:
        try:
            input("\n⏎ Press Enter when a customer approaches (Ctrl+C to exit)...")
            
            # Reset conversation history for new customer
            reset_conversation()
            
            # Analyze person if camera is enabled
            person_traits = {}
            if use_camera:
                person_traits = analyze_person(duration_seconds=3.0)
            
            # First turn: AI initiates with personalized greeting
            run_conversation_turn(duration=5.0, is_first_turn=True, person_traits=person_traits)
            
            # Continue conversation automatically - no manual input needed
            for _ in range(5):  # Max 5 more exchanges
                run_conversation_turn(duration=5.0, is_first_turn=False)
            
            print("\n👋 Conversation complete! Ready for next customer.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def main():
    """Main entry point."""
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Check for flags
    use_camera = "--no-camera" not in sys.argv
    
    # Check for loop mode
    if "--loop" in sys.argv:
        run_conversation_loop(use_camera=use_camera)
    else:
        print("\n🏪 STOREFRONT AI ASSISTANT")
        print("Tip: Use --loop for continuous customer mode")
        print("Tip: Use --no-camera to skip person detection\n")
        
        # Reset conversation for fresh start
        reset_conversation()
        
        # Analyze person if camera is enabled
        person_traits = {}
        if use_camera:
            person_traits = analyze_person(duration_seconds=3.0)
        
        # Run full conversation automatically
        run_conversation_turn(duration=5.0, is_first_turn=True, person_traits=person_traits)
        
        # Continue conversation - AI and user take turns
        for _ in range(5):  # Max 5 more exchanges
            run_conversation_turn(duration=5.0, is_first_turn=False)
        
        print("\n👋 Conversation complete!")


if __name__ == "__main__":
    main()
