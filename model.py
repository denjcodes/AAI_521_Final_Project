import numpy as np
import mediapipe as mp
from typing import Optional, Tuple

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class ASLDetector:
    """ASL hand gesture detection using MediaPipe Hands."""

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
        """
        Process a single frame for hand detection and ASL classification.

        Args:
            image: RGB image array

        Returns:
            Tuple of (annotated_image, predicted_letter, confidence)
        """
        results = self.hands.process(image)

        if not results.multi_hand_landmarks:
            return image, None, None

        annotated_image = image.copy()

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            letter, confidence = self._classify_gesture(hand_landmarks)

            return annotated_image, letter, confidence

        return annotated_image, None, None

    def _classify_gesture(self, landmarks) -> Tuple[str, float]:
        """
        Classify ASL gesture based on hand landmarks.

        Args:
            landmarks: MediaPipe hand landmarks

        Returns:
            Tuple of (predicted_letter, confidence)
        """
        landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

        thumb_tip = landmark_array[4]
        index_tip = landmark_array[8]
        middle_tip = landmark_array[12]
        ring_tip = landmark_array[16]
        pinky_tip = landmark_array[20]

        thumb_ip = landmark_array[3]
        index_pip = landmark_array[6]
        middle_pip = landmark_array[10]
        ring_pip = landmark_array[14]
        pinky_pip = landmark_array[18]

        wrist = landmark_array[0]

        fingers_extended = [
            thumb_tip[0] > thumb_ip[0] if thumb_tip[0] > wrist[0] else thumb_tip[0] < thumb_ip[0],
            index_tip[1] < index_pip[1],
            middle_tip[1] < middle_pip[1],
            ring_tip[1] < ring_pip[1],
            pinky_tip[1] < pinky_pip[1]
        ]

        num_extended = sum(fingers_extended[1:])

        if num_extended == 0 and not fingers_extended[0]:
            return "A", 0.8
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            return "V", 0.85
        elif all(fingers_extended[1:]):
            if fingers_extended[0]:
                return "B", 0.8
            else:
                return "4", 0.75
        elif fingers_extended[1] and not any(fingers_extended[2:]):
            return "1", 0.8
        elif num_extended == 3 and fingers_extended[1] and fingers_extended[2] and fingers_extended[3]:
            return "W", 0.75
        else:
            return "Unknown", 0.5

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
