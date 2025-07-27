"""
Camera Management Module
=======================

Camera initialization, configuration, and frame capture with error handling.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CameraManager:
    """Camera management with error handling and fallback mechanisms"""
    
    def __init__(self, device_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self.camera = None
        self.is_available = False
        
        self._init_camera()
    
    def _init_camera(self):
        """Initialize camera with error handling"""
        try:
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                logger.warning(f"Failed to open camera {self.device_id}")
                # Try default camera
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    logger.error("Failed to open any camera")
                    self.is_available = False
                    return
            
            # Set camera properties for high performance
            width, height = self.resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Additional settings for better performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Test camera by reading a frame
            ret, test_frame = self.camera.read()
            if ret:
                self.is_available = True
                actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Camera initialized successfully: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            else:
                logger.error("Camera opened but failed to read frame")
                self.is_available = False
                
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self.is_available = False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera with error handling"""
        if not self.is_available or self.camera is None:
            return False, None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return False, None
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading camera frame: {e}")
            return False, None
    
    def get_placeholder_frame(self) -> np.ndarray:
        """Get placeholder frame when camera is not available"""
        width, height = self.resolution
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add text
        cv2.putText(placeholder, "CAMERA NOT AVAILABLE", 
                   (width//2 - 150, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(placeholder, "Check camera connection", 
                   (width//2 - 120, height//2 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return placeholder
    
    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.is_available = False
            logger.info("Camera released")
    
    def is_camera_available(self) -> bool:
        """Check if camera is available and working"""
        return self.is_available
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        if not self.is_available or self.camera is None:
            return {"status": "unavailable"}
        
        try:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            return {
                "status": "available",
                "device_id": self.device_id,
                "resolution": (width, height),
                "fps": fps
            }
        except Exception as e:
            logger.error(f"Error getting camera info: {e}")
            return {"status": "error"} 