"""
Gesture Recognition Module
=========================

Hand gesture recognition using MediaPipe with fallback mechanisms.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class GestureController:
    """Hand gesture recognition for navigation control"""
    
    def __init__(self, enable_mediapipe: bool = True):
        self.enabled = False
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
        
        # Gesture commands mapping
        self.gestures = {
            'stop': self._detect_stop_gesture,
            'forward': self._detect_forward_gesture,
            'left': self._detect_left_gesture,
            'right': self._detect_right_gesture,
            'help': self._detect_help_gesture
        }
        
        if enable_mediapipe:
            self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe with error handling"""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.enabled = True
            logger.info("MediaPipe gesture recognition initialized successfully")
            
        except ImportError as e:
            logger.warning(f"MediaPipe not available (Python 3.13 compatibility issue), gesture recognition disabled: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            self.enabled = False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for gesture recognition"""
        gesture_data = {
            'detected': False,
            'gesture': None,
            'confidence': 0.0,
            'landmarks': None
        }
        
        if not self.enabled or self.hands is None:
            return gesture_data
        
        try:
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect gestures
                    gesture_data['landmarks'] = hand_landmarks
                    for gesture_name, detector in self.gestures.items():
                        confidence = detector(hand_landmarks)
                        if confidence > 0.8:
                            gesture_data.update({
                                'detected': True,
                                'gesture': gesture_name,
                                'confidence': confidence
                            })
                            break
                    
                    # Only process first hand
                    break
            
            return gesture_data
            
        except Exception as e:
            logger.error(f"Gesture recognition failed: {e}")
            return gesture_data
    
    def _detect_stop_gesture(self, landmarks) -> float:
        """Detect stop gesture (open palm)"""
        try:
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            middle_tip = landmarks.landmark[12]
            ring_tip = landmarks.landmark[16]
            pinky_tip = landmarks.landmark[20]
            
            # Check if all fingertips are extended
            fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
            extended = all(tip.y < landmarks.landmark[0].y for tip in fingertips)
            
            if extended and thumb_tip.x < landmarks.landmark[0].x:
                return 0.95
            return 0.0
            
        except Exception as e:
            logger.error(f"Stop gesture detection failed: {e}")
            return 0.0
    
    def _detect_forward_gesture(self, landmarks) -> float:
        """Detect forward gesture (pointing)"""
        try:
            index_tip = landmarks.landmark[8]
            index_pip = landmarks.landmark[6]
            
            # Check if index finger is extended and others are closed
            if index_tip.y < index_pip.y:
                other_fingers = [
                    landmarks.landmark[i] for i in [12, 16, 20]
                ]
                if all(tip.y > landmarks.landmark[i].y for i, tip in zip([9, 13, 17], other_fingers)):
                    return 0.90
            return 0.0
            
        except Exception as e:
            logger.error(f"Forward gesture detection failed: {e}")
            return 0.0
    
    def _detect_left_gesture(self, landmarks) -> float:
        """Detect left turn gesture"""
        try:
            palm_base = landmarks.landmark[0]
            thumb_tip = landmarks.landmark[4]
            pinky_tip = landmarks.landmark[20]
            
            if thumb_tip.x < palm_base.x and pinky_tip.x < palm_base.x:
                return 0.85
            return 0.0
            
        except Exception as e:
            logger.error(f"Left gesture detection failed: {e}")
            return 0.0
    
    def _detect_right_gesture(self, landmarks) -> float:
        """Detect right turn gesture"""
        try:
            palm_base = landmarks.landmark[0]
            thumb_tip = landmarks.landmark[4]
            pinky_tip = landmarks.landmark[20]
            
            if thumb_tip.x > palm_base.x and pinky_tip.x > palm_base.x:
                return 0.85
            return 0.0
            
        except Exception as e:
            logger.error(f"Right gesture detection failed: {e}")
            return 0.0
    
    def _detect_help_gesture(self, landmarks) -> float:
        """Detect help gesture (raised hand)"""
        try:
            wrist = landmarks.landmark[0]
            middle_tip = landmarks.landmark[12]
            
            if middle_tip.y < wrist.y:
                return 0.92
            return 0.0
            
        except Exception as e:
            logger.error(f"Help gesture detection failed: {e}")
            return 0.0
    
    def is_enabled(self) -> bool:
        """Check if gesture recognition is enabled"""
        return self.enabled 