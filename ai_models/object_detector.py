"""
Enhanced Object Detection Module
===============================

Advanced object detection with indoor navigation specific features.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
import time
from datetime import datetime

from models.data_models import DetectionResult

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Enhanced object detector with indoor navigation capabilities"""
    
    def __init__(self, model_path: str = "yolo11n.pt", confidence_threshold: float = 0.5):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Indoor navigation specific object mappings
        self.indoor_objects = {
            # YOLO COCO classes that map to indoor features
            'person': 'person',
            'chair': 'seating',
            'couch': 'seating', 
            'bed': 'furniture',
            'dining table': 'table',
            'tv': 'electronics',
            'laptop': 'electronics',
            'cell phone': 'electronics',
            'book': 'reading_material',
            'cup': 'container',
            'bottle': 'container',
            'bowl': 'container',
            'fork': 'utensil',
            'knife': 'utensil',
            'spoon': 'utensil',
            'door': 'exit_entrance',
            'window': 'exit_entrance',
            'stairs': 'stairs',
            'elevator': 'elevator',  # Custom detection
            'toilet': 'toilet',      # Custom detection
            'sink': 'bathroom_fixture',
            'mirror': 'bathroom_fixture',
            'fire extinguisher': 'safety_equipment',
            'exit sign': 'exit_sign',
            'emergency exit': 'emergency_exit'
        }
        
        # Risk levels for different objects
        self.risk_levels = {
            'person': 'medium',
            'stairs': 'high',
            'elevator': 'medium',
            'toilet': 'low',
            'exit_entrance': 'low',
            'emergency_exit': 'low',
            'safety_equipment': 'low',
            'exit_sign': 'low',
            'furniture': 'medium',
            'electronics': 'medium',
            'seating': 'low',
            'table': 'medium',
            'container': 'low',
            'utensil': 'low',
            'reading_material': 'low',
            'bathroom_fixture': 'low'
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect objects in frame with enhanced indoor navigation features"""
        detections = []
        
        if self.model is None:
            return detections
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = result.names[class_id]
                        
                        # Map to indoor navigation object type
                        object_type = self._map_to_indoor_object(class_name)
                        
                        # Calculate distance (simplified - would need depth sensor for accuracy)
                        distance = self._estimate_distance(x1, y1, x2, y2, frame.shape)
                        
                        # Get risk level
                        risk_level = self.risk_levels.get(object_type, 'medium')
                        
                        # Generate navigation advice
                        navigation_advice = self._generate_navigation_advice(object_type, distance, risk_level)
                        
                        # Create detection result
                        detection = DetectionResult(
                            object_type=object_type,
                            confidence=confidence,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            distance=distance,
                            depth_map=np.array([]),  # Will be filled by depth estimator if available
                            timestamp=datetime.now(),
                            risk_level=risk_level,
                            navigation_advice=navigation_advice
                        )
                        
                        detections.append(detection)
            
            # Add custom indoor feature detection
            custom_detections = self._detect_custom_features(frame)
            detections.extend(custom_detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return detections
    
    def _map_to_indoor_object(self, class_name: str) -> str:
        """Map YOLO class names to indoor navigation objects"""
        return self.indoor_objects.get(class_name, class_name)
    
    def _estimate_distance(self, x1: float, y1: float, x2: float, y2: float, frame_shape: tuple) -> float:
        """Estimate distance to object based on bounding box size"""
        # Simplified distance estimation
        # In a real system, you'd use depth sensors or stereo vision
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        
        # Larger objects appear closer
        relative_size = bbox_area / frame_area
        
        # Rough distance estimation (in meters)
        if relative_size > 0.1:
            return 0.5  # Very close
        elif relative_size > 0.05:
            return 1.0  # Close
        elif relative_size > 0.02:
            return 2.0  # Medium distance
        else:
            return 3.0  # Far
    
    def _generate_navigation_advice(self, object_type: str, distance: float, risk_level: str) -> str:
        """Generate navigation advice based on detected object"""
        if risk_level == 'high':
            if object_type == 'stairs':
                return f"⚠️ Stairs detected {distance:.1f}m ahead - Use handrail"
            elif object_type == 'person':
                return f"👤 Person {distance:.1f}m ahead - Give space"
            else:
                return f"⚠️ {object_type.title()} {distance:.1f}m ahead - Proceed with caution"
        
        elif risk_level == 'medium':
            if object_type == 'elevator':
                return f"🛗 Elevator {distance:.1f}m ahead - Check if operational"
            elif object_type == 'furniture':
                return f"🪑 {object_type.title()} {distance:.1f}m ahead - Navigate around"
            else:
                return f"📍 {object_type.title()} {distance:.1f}m ahead"
        
        else:  # low risk
            if object_type == 'toilet':
                return f"🚽 Toilet {distance:.1f}m ahead"
            elif object_type == 'exit_entrance':
                return f"🚪 {object_type.title()} {distance:.1f}m ahead"
            elif object_type == 'exit_sign':
                return f"🚪 Exit sign {distance:.1f}m ahead"
            else:
                return f"📍 {object_type.title()} {distance:.1f}m ahead"
    
    def _detect_custom_features(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect custom indoor features using computer vision techniques"""
        custom_detections = []
        
        try:
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Detect exit signs (red rectangles with text)
            exit_signs = self._detect_exit_signs(frame)
            custom_detections.extend(exit_signs)
            
            # 2. Detect elevator buttons (circular buttons)
            elevator_buttons = self._detect_elevator_buttons(gray)
            custom_detections.extend(elevator_buttons)
            
            # 3. Detect bathroom symbols (universal symbols)
            bathroom_symbols = self._detect_bathroom_symbols(gray)
            custom_detections.extend(bathroom_symbols)
            
            # 4. Detect stairs (horizontal lines pattern)
            stairs = self._detect_stairs(gray)
            custom_detections.extend(stairs)
            
        except Exception as e:
            logger.error(f"Custom feature detection failed: {e}")
        
        return custom_detections
    
    def _detect_exit_signs(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect exit signs using color and shape detection"""
        detections = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define red color range for exit signs
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for red regions
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's roughly rectangular (exit sign shape)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 2.0:
                        distance = self._estimate_distance(x, y, x+w, y+h, frame.shape)
                        
                        detection = DetectionResult(
                            object_type='exit_sign',
                            confidence=0.7,
                            bbox=(x, y, x+w, y+h),
                            distance=distance,
                            depth_map=np.array([]),  # Will be filled by depth estimator if available
                            timestamp=datetime.now(),
                            risk_level='low',
                            navigation_advice=f"🚪 Exit sign {distance:.1f}m ahead"
                        )
                        detections.append(detection)
        
        except Exception as e:
            logger.error(f"Exit sign detection failed: {e}")
        
        return detections
    
    def _detect_elevator_buttons(self, gray: np.ndarray) -> List[DetectionResult]:
        """Detect elevator buttons using circle detection"""
        detections = []
        
        try:
            # Detect circles (elevator buttons)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=50, param2=30, minRadius=10, maxRadius=50
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    distance = self._estimate_distance(x-r, y-r, x+r, y+r, gray.shape)
                    
                    detection = DetectionResult(
                        object_type='elevator_button',
                        confidence=0.6,
                        bbox=(x-r, y-r, x+r, y+r),
                        distance=distance,
                        depth_map=np.array([]),  # Will be filled by depth estimator if available
                        timestamp=datetime.now(),
                        risk_level='low',
                        navigation_advice=f"🛗 Elevator button {distance:.1f}m ahead"
                    )
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"Elevator button detection failed: {e}")
        
        return detections
    
    def _detect_bathroom_symbols(self, gray: np.ndarray) -> List[DetectionResult]:
        """Detect bathroom symbols using template matching"""
        detections = []
        
        try:
            # Simple edge detection for symbol recognition
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours that might be bathroom symbols
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:  # Typical bathroom symbol size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's roughly square (bathroom symbols are often square)
                    aspect_ratio = w / h
                    if 0.8 < aspect_ratio < 1.2:
                        distance = self._estimate_distance(x, y, x+w, y+h, gray.shape)
                        
                        detection = DetectionResult(
                            object_type='bathroom_symbol',
                            confidence=0.5,
                            bbox=(x, y, x+w, y+h),
                            distance=distance,
                            depth_map=np.array([]),  # Will be filled by depth estimator if available
                            timestamp=datetime.now(),
                            risk_level='low',
                            navigation_advice=f"🚽 Bathroom {distance:.1f}m ahead"
                        )
                        detections.append(detection)
        
        except Exception as e:
            logger.error(f"Bathroom symbol detection failed: {e}")
        
        return detections
    
    def _detect_stairs(self, gray: np.ndarray) -> List[DetectionResult]:
        """Detect stairs using line detection"""
        detections = []
        
        try:
            # Detect horizontal lines (stair edges)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                horizontal_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                    
                    # Check if line is roughly horizontal (stair edge)
                    if abs(angle) < 20:
                        horizontal_lines += 1
                
                # If we detect multiple horizontal lines, it might be stairs
                if horizontal_lines >= 3:
                    # Estimate stairs location (bottom of frame)
                    h, w = gray.shape
                    x, y, w_bbox, h_bbox = 0, h//2, w, h//2
                    
                    distance = self._estimate_distance(x, y, x+w_bbox, y+h_bbox, gray.shape)
                    
                    detection = DetectionResult(
                        object_type='stairs',
                        confidence=0.7,
                        bbox=(x, y, x+w_bbox, y+h_bbox),
                        distance=distance,
                        depth_map=np.array([]),  # Will be filled by depth estimator if available
                        timestamp=datetime.now(),
                        risk_level='high',
                        navigation_advice=f"⚠️ Stairs {distance:.1f}m ahead - Use handrail"
                    )
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"Stairs detection failed: {e}")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection results on frame with enhanced visualization"""
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on risk level
            if detection.risk_level == 'high':
                color = (0, 0, 255)  # Red
            elif detection.risk_level == 'medium':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.object_type.title()} {detection.distance:.1f}m"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence
            conf_text = f"Conf: {detection.confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to: {threshold}") 