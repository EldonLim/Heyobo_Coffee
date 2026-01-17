"""Hand gesture control for Coffee Game using MediaPipe."""

import cv2
import numpy as np
import os
from typing import Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandController:
    """Controls game movement using hand gestures detected via webcam."""

    def __init__(self, detection_confidence: float = 0.5, max_hands: int = 1):
        """
        Initialize the hand controller.

        Args:
            detection_confidence: Confidence threshold for hand detection (0-1)
            max_hands: Maximum number of hands to detect
        """
        self.video: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._last_finger_count = 0
        
        # Get path to model file
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Please download it from: "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        
        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def start(self, camera_index: int = 0) -> bool:
        """
        Start the video capture.

        Args:
            camera_index: Index of the camera to use

        Returns:
            True if camera started successfully, False otherwise
        """
        self.video = cv2.VideoCapture(camera_index)
        if not self.video.isOpened():
            print("Error: Could not open camera")
            return False
        self.is_running = True
        return True

    def stop(self) -> None:
        """Stop the video capture and close windows."""
        self.is_running = False
        if self.video is not None:
            self.video.release()
        cv2.destroyAllWindows()

    def _fingers_up(self, hand_landmarks, handedness: str) -> list:
        """
        Detect which fingers are up (extended) - matches cvzone's fingersUp() method.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right" hand
            
        Returns:
            List of 5 values [thumb, index, middle, ring, pinky], 1=up, 0=down
        """
        landmarks = hand_landmarks
        fingers = []
        
        # Tip landmark indices for each finger
        tip_ids = [4, 8, 12, 16, 20]
        
        # Thumb - check x position relative to IP joint (landmark 3)
        # Note: After horizontal flip, handedness appears reversed in the image
        if handedness == "Right":
            # Right hand: thumb tip x > IP x means thumb is up
            if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Left hand: thumb tip x < IP x means thumb is up
            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # 4 Fingers - check if tip y < PIP y (tip is above PIP when extended)
        # tip_ids[1] = 8 (index), check against 6 (8-2)
        # tip_ids[2] = 12 (middle), check against 10
        # tip_ids[3] = 16 (ring), check against 14
        # tip_ids[4] = 20 (pinky), check against 18
        for i in range(1, 5):
            if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    def _count_fingers(self, hand_landmarks, handedness: str = "Right") -> int:
        """
        Count the number of extended fingers based on hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right" hand
            
        Returns:
            Number of extended fingers (0-5)
        """
        fingers = self._fingers_up(hand_landmarks, handedness)
        return sum(fingers)

    def get_finger_count(self) -> Tuple[Optional[int], Optional[any]]:
        """
        Get the current finger count from the webcam.

        Returns:
            Tuple of (finger_count, frame) or (None, None) if no hand detected
        """
        if self.video is None or not self.is_running:
            return None, None

        ret, frame = self.video.read()
        if not ret:
            return None, None

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        detection_result = self.detector.detect(mp_image)

        finger_count = None
        fingers_state = None
        
        if detection_result.hand_landmarks:
            # Get first hand's landmarks and handedness
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Get handedness (Left/Right)
            handedness = "Right"  # Default
            if detection_result.handedness and len(detection_result.handedness) > 0:
                handedness = detection_result.handedness[0][0].category_name
            
            fingers_state = self._fingers_up(hand_landmarks, handedness)
            finger_count = sum(fingers_state)
            self._last_finger_count = finger_count
            
            # Draw landmarks on frame
            self._draw_landmarks(frame, hand_landmarks)
            
            # Print fingers state for debugging (like cvzone does)
            print(f"fingersUp: {fingers_state}")
            
        if finger_count is not None:
            self._draw_ui(frame, finger_count)

        return finger_count, frame

    def _draw_landmarks(self, frame, hand_landmarks) -> None:
        """Draw hand landmarks on the frame."""
        height, width = frame.shape[:2]
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),  # Palm
        ]
        
        for start_idx, end_idx in connections:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmark points
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    def _draw_ui(self, frame, finger_count: int) -> None:
        """Draw UI elements on the frame."""
        height, width = frame.shape[:2]

        # Draw bottom info bars
        cv2.rectangle(frame, (0, height), (200, height - 55), (50, 50, 255), -1)
        cv2.rectangle(frame, (width, height), (width - 200, height - 55), (50, 50, 255), -1)

        # Finger count text
        cv2.putText(
            frame,
            f'Fingers: {finger_count}',
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Direction text
        if finger_count % 2 == 1:  # Odd = left
            direction = "LEFT <-"
            color = (0, 255, 255)  # Yellow
        else:  # Even = right
            direction = "-> RIGHT"
            color = (0, 255, 0)  # Green

        cv2.putText(
            frame,
            direction,
            (width - 180, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    def get_direction(self) -> Optional[str]:
        """
        Get movement direction based on finger count.

        Returns:
            'left' for odd finger count, 'right' for even, None if no hand detected
        """
        finger_count, _ = self.get_finger_count()

        if finger_count is None:
            return None

        if finger_count % 2 == 1:  # Odd = left
            return 'left'
        else:  # Even = right
            return 'right'

    def show_frame(self, frame) -> bool:
        """
        Display the camera frame with UI.

        Args:
            frame: The frame to display

        Returns:
            False if 'q' was pressed to quit, True otherwise
        """
        if frame is not None:
            cv2.imshow("Hand Control - Coffee Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        return True


