"""
Gaze detection module to detect when user is looking at the camera.
Uses MediaPipe Face Landmarker (Tasks API) to track eye landmarks and determine gaze direction.
"""

import cv2
import numpy as np
import time
import os
from typing import Optional, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GazeDetector:
    """
    Detects when user is looking at the camera for a specified duration.
    
    Usage:
        detector = GazeDetector(required_duration=2.0)
        
        # Option 1: Blocking wait
        if detector.wait_for_gaze():
            print("User is looking at camera!")
        
        # Option 2: With callback
        detector.start_detection(on_gaze_detected=my_callback)
    """
    
    # Eye landmark indices for MediaPipe Face Landmarker
    # Left eye
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_IRIS_CENTER = 468  # Iris landmarks (468-472 for left)
    
    # Right eye
    RIGHT_EYE_OUTER = 362
    RIGHT_EYE_INNER = 263
    RIGHT_IRIS_CENTER = 473  # Iris landmarks (473-477 for right)
    
    def __init__(
        self,
        required_duration: float = 2.0,
        camera_index: int = 0,
        gaze_threshold: float = 0.35,
        show_preview: bool = True,
        model_path: Optional[str] = None
    ):
        """
        Initialize the gaze detector.
        
        Args:
            required_duration: Seconds user must look at camera to trigger detection
            camera_index: Camera device index
            gaze_threshold: How centered the iris must be (0-0.5, lower = stricter)
            show_preview: Whether to show camera preview window
            model_path: Path to face_landmarker.task model file
        """
        self.required_duration = required_duration
        self.camera_index = camera_index
        self.gaze_threshold = gaze_threshold
        self.show_preview = show_preview
        
        # State tracking
        self.gaze_start_time: Optional[float] = None
        self.is_looking = False
        self.detection_complete = False
        
        # Find model path
        if model_path is None:
            # Look for model in ai_agent folder or current directory
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "ai_agent", "face_landmarker.task"),
                os.path.join(os.path.dirname(__file__), "face_landmarker.task"),
                "face_landmarker.task"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "face_landmarker.task not found. Download from: "
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                )
        
        # Create Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    def _calculate_iris_position(
        self,
        landmarks,
        eye_outer_idx: int,
        eye_inner_idx: int,
        iris_center_idx: int,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float]:
        """
        Calculate normalized iris position within the eye.
        
        Returns:
            Tuple of (horizontal_ratio, vertical_ratio, eye_width)
            horizontal_ratio: 0 = looking left, 0.5 = center, 1 = looking right
        """
        # Get eye corner positions
        outer = landmarks[eye_outer_idx]
        inner = landmarks[eye_inner_idx]
        iris = landmarks[iris_center_idx]
        
        # Convert to pixel coordinates
        outer_x = outer.x * img_width
        inner_x = inner.x * img_width
        iris_x = iris.x * img_width
        iris_y = iris.y * img_height
        
        # Calculate eye width
        eye_width = abs(inner_x - outer_x)
        
        if eye_width < 1:
            return 0.5, 0.5, 0
        
        # Calculate horizontal position (0-1, 0.5 is center)
        min_x = min(outer_x, inner_x)
        horizontal_ratio = (iris_x - min_x) / eye_width
        
        # Get vertical bounds (using upper and lower eyelid landmarks)
        # For simplicity, we use the eye height estimation
        eye_center_y = (outer.y + inner.y) / 2 * img_height
        vertical_ratio = 0.5  # Simplified - just check horizontal for now
        
        return horizontal_ratio, vertical_ratio, eye_width
    
    def _is_looking_at_camera(self, landmarks, img_width: int, img_height: int) -> bool:
        """
        Determine if the user is looking at the camera based on iris position.
        """
        # Calculate iris positions for both eyes
        left_h, left_v, left_w = self._calculate_iris_position(
            landmarks,
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.LEFT_IRIS_CENTER,
            img_width,
            img_height
        )
        
        right_h, right_v, right_w = self._calculate_iris_position(
            landmarks,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_INNER,
            self.RIGHT_IRIS_CENTER,
            img_width,
            img_height
        )
        
        # Check if eyes are detected (have reasonable width)
        if left_w < 5 or right_w < 5:
            return False
        
        # Check if both irises are centered (looking at camera)
        # Centered means horizontal ratio is close to 0.5
        left_centered = abs(left_h - 0.5) < self.gaze_threshold
        right_centered = abs(right_h - 0.5) < self.gaze_threshold
        
        return left_centered and right_centered
    
    def _draw_debug_info(
        self,
        frame: np.ndarray,
        is_looking: bool,
        elapsed_time: float,
        landmarks=None
    ) -> np.ndarray:
        """Draw debug visualization on frame."""
        h, w = frame.shape[:2]
        
        # Draw status
        status_color = (0, 255, 0) if is_looking else (0, 0, 255)
        status_text = "Looking at camera" if is_looking else "Not looking"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, status_color, 2)
        
        # Draw progress bar
        if is_looking and elapsed_time > 0:
            progress = min(elapsed_time / self.required_duration, 1.0)
            bar_width = int(w * 0.6)
            bar_height = 20
            bar_x = (w - bar_width) // 2
            bar_y = h - 50
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)
            
            # Progress fill
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         (0, 255, 0), -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)
            
            # Time text
            time_text = f"{elapsed_time:.1f}s / {self.required_duration:.1f}s"
            text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, time_text, (text_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instruction
        instruction = "Look at the camera to start"
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, instruction, (text_x, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw eye landmarks if available
        if landmarks:
            for idx in [self.LEFT_IRIS_CENTER, self.RIGHT_IRIS_CENTER]:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        return frame
    
    def wait_for_gaze(self, timeout: Optional[float] = None) -> bool:
        """
        Block until user looks at camera for required duration.
        
        Args:
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            True if gaze detected, False if timeout or cancelled
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("❌ Could not open camera for gaze detection")
            return False
        
        self.gaze_start_time = None
        self.detection_complete = False
        start_wait_time = time.time()
        
        print("👁️ Waiting for user to look at camera...")
        
        try:
            while not self.detection_complete:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe Image and detect
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.face_landmarker.detect(mp_image)
                
                # Check gaze
                is_looking = False
                landmarks = None
                
                if results.face_landmarks and len(results.face_landmarks) > 0:
                    landmarks = results.face_landmarks[0]
                    is_looking = self._is_looking_at_camera(landmarks, w, h)
                
                # Update timing
                elapsed_time = 0
                if is_looking:
                    if self.gaze_start_time is None:
                        self.gaze_start_time = time.time()
                    elapsed_time = time.time() - self.gaze_start_time
                    
                    if elapsed_time >= self.required_duration:
                        self.detection_complete = True
                        print(f"✅ Gaze detected for {self.required_duration}s!")
                else:
                    self.gaze_start_time = None
                
                # Check timeout
                if timeout and (time.time() - start_wait_time) > timeout:
                    print("⏰ Gaze detection timed out")
                    break
                
                # Show preview if enabled
                if self.show_preview:
                    debug_frame = self._draw_debug_info(frame, is_looking, elapsed_time, landmarks)
                    cv2.imshow("Gaze Detection - Look at camera", debug_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        print("❌ Gaze detection cancelled")
                        break
            
        finally:
            cap.release()
            if self.show_preview:
                cv2.destroyWindow("Gaze Detection - Look at camera")
        
        return self.detection_complete
    
    def reset(self):
        """Reset detection state."""
        self.gaze_start_time = None
        self.is_looking = False
        self.detection_complete = False
    
    def cleanup(self):
        """Release resources."""
        self.face_landmarker.close()


def test_gaze_detector():
    """Test the gaze detector standalone."""
    print("=" * 50)
    print("Gaze Detection Test")
    print("Look at the camera for 2 seconds to trigger detection")
    print("Press 'q' or ESC to quit")
    print("=" * 50)
    
    detector = GazeDetector(required_duration=2.0, show_preview=True)
    
    result = detector.wait_for_gaze(timeout=30)
    
    if result:
        print("\n🎉 Success! You looked at the camera!")
    else:
        print("\n😔 Detection failed or was cancelled")
    
    detector.cleanup()


if __name__ == "__main__":
    test_gaze_detector()
