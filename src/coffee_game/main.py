"""Main entry point for the Coffee Game."""

import argparse
import threading
from .game import CoffeeGame


def wait_for_user_attention(gaze_duration: float = 2.0, timeout: float = 60.0, show_preview: bool = True) -> bool:
    """
    Wait for user to look at the camera before starting.
    
    Args:
        gaze_duration: How long user must look at camera (seconds)
        timeout: Maximum time to wait for gaze (seconds)
        show_preview: Whether to show camera preview (disable if using Tkinter after)
        
    Returns:
        True if user looked at camera, False if timeout/cancelled
    """
    try:
        from .gaze_detector import GazeDetector
        import cv2
        import time
        
        print("\n" + "="*50)
        print("👁️ ATTENTION CHECK")
        print("Look at the camera for 2 seconds to begin!")
        print("="*50 + "\n")
        
        detector = GazeDetector(
            required_duration=gaze_duration,
            show_preview=show_preview
        )
        
        result = detector.wait_for_gaze(timeout=timeout)
        detector.cleanup()
        
        # Ensure all OpenCV windows are fully destroyed before Tkinter starts
        cv2.destroyAllWindows()
        # Process any pending window events
        for _ in range(10):
            cv2.waitKey(1)
        # Delay to let macOS clean up GUI resources
        time.sleep(0.5)
        
        return result
        
    except ImportError as e:
        print(f"⚠️ Gaze detection not available: {e}")
        print("Skipping gaze detection...")
        return True  # Continue anyway if gaze detection unavailable


def run_with_ai_assistant(
    use_hand_control: bool = False,
    use_camera: bool = True,
    use_heyobo: bool = False,
    use_gaze: bool = False
) -> None:
    """
    Run the game with AI assistant introduction.
    
    The AI assistant will have a conversation with the user first,
    and when they agree to play, the game will start.
    
    Args:
        use_hand_control: Enable hand gesture control for the game
        use_camera: Use camera for person detection
        use_heyobo: Show HeyoBo animated character UI
        use_gaze: Require user to look at camera before starting
    """
    # Wait for user attention if gaze detection is enabled
    # Disable preview when HeyoBo is enabled to avoid OpenCV/Tkinter conflict on macOS
    if use_gaze:
        if not wait_for_user_attention(show_preview=not use_heyobo):
            print("\n👋 Come back when you're ready to chat!")
            return
    
    try:
        from .ai_agent.init import run_ai_assistant, set_heyobo
        
        heyobo = None
        
        if use_heyobo:
            try:
                from .heyobo import HeyoBo
                import os
                
                # Get assets path relative to this file
                assets_path = os.path.join(os.path.dirname(__file__), "assets")
                
                # Create HeyoBo instance
                heyobo = HeyoBo(assets_path=assets_path, scale=0.5)
                set_heyobo(heyobo)
                
                # Run AI conversation in a separate thread
                result = {"user_wants_to_play": False}
                
                def ai_thread():
                    result["user_wants_to_play"] = run_ai_assistant(use_camera=use_camera)
                    # Close HeyoBo window after conversation
                    if heyobo:
                        heyobo.root.after(100, heyobo.root.quit)
                
                thread = threading.Thread(target=ai_thread, daemon=True)
                thread.start()
                
                # Run Tkinter main loop in main thread
                heyobo.run()
                
                # Wait for thread to finish
                thread.join(timeout=1.0)
                
                user_wants_to_play = result["user_wants_to_play"]
                
            except ImportError as e:
                print(f"⚠️ Could not load HeyoBo: {e}")
                print("Running without HeyoBo UI...")
                user_wants_to_play = run_ai_assistant(use_camera=use_camera)
        else:
            # Run AI assistant conversation without HeyoBo
            user_wants_to_play = run_ai_assistant(use_camera=use_camera)
        
        if user_wants_to_play:
            print("\n" + "="*50)
            print("🎮 STARTING COFFEE CATCHER GAME!")
            print("="*50)
            
            # Run the game
            game = CoffeeGame(use_hand_control=use_hand_control)
            result = game.run()
            game.quit()
            
            # Print results
            voucher = get_voucher_percentage(result['fullness'])
            print("\n" + "="*50)
            print(f"☕ You filled {result['fullness']}% of the cup!")
            print(f"🎉 You won a {voucher}% discount voucher!")
            print("="*50 + "\n")
        else:
            print("\n👋 Maybe next time! Come back for coffee!")
            
    except ImportError as e:
        print(f"❌ Could not import AI assistant module: {e}")
        print("Running game directly without AI assistant...")
        game = CoffeeGame(use_hand_control=use_hand_control)
        result = game.run()
        game.quit()


def get_voucher_percentage(fullness: int) -> int:
    """Get voucher percentage based on cup fullness."""
    if fullness >= 90:
        return 20
    elif fullness >= 60:
        return 15
    elif fullness >= 30:
        return 10
    else:
        return 5


def main() -> None:
    """Run the coffee game."""
    parser = argparse.ArgumentParser(description="Coffee Catcher Game")
    parser.add_argument(
        "--hand-control", "-hc",
        action="store_true",
        help="Enable hand gesture control via webcam (requires opencv-python and mediapipe)"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI assistant and start game directly"
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable camera for AI person detection (uses generic greeting)"
    )
    parser.add_argument(
        "--heyobo",
        action="store_true",
        help="Show HeyoBo animated character during AI conversation"
    )
    parser.add_argument(
        "--gaze", "-g",
        action="store_true",
        help="Require user to look at camera for 2 seconds before starting"
    )
    args = parser.parse_args()

    if args.no_ai:
        # Run game directly without AI assistant
        game = CoffeeGame(use_hand_control=args.hand_control)
        result = game.run()
        game.quit()
        
        voucher = get_voucher_percentage(result['fullness'])
        print("\n" + "="*50)
        print(f"☕ You filled {result['fullness']}% of the cup!")
        print(f"🎉 You won a {voucher}% discount voucher!")
        print("="*50 + "\n")
    else:
        # Run with AI assistant
        run_with_ai_assistant(
            use_hand_control=args.hand_control,
            use_camera=not args.no_camera,
            use_heyobo=args.heyobo,
            use_gaze=args.gaze
        )


if __name__ == "__main__":
    main()
