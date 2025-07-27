"""
Depth Estimation Module
======================

Advanced depth estimation using MiDaS model with fallback mechanisms.
"""

import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Advanced depth estimation using MiDaS model with fallback"""
    
    def __init__(self, enable_midas: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enabled = False
        self.model = None
        self.transform = None
        
        if enable_midas:
            self._init_midas()
    
    def _init_midas(self):
        """Initialize MiDaS model with proper error handling"""
        try:
            # Try to import timm for MiDaS
            import timm
            
            # Load MiDaS model for depth estimation
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = self.transforms.small_transform
            
            self.enabled = True
            logger.info("MiDaS depth estimation initialized successfully")
            
        except ImportError as e:
            logger.warning(f"timm not available, depth estimation disabled: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize MiDaS depth estimation: {e}")
            self.enabled = False
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from RGB frame"""
        if not self.enabled or self.model is None:
            return self._fallback_depth_estimation(frame)
        
        try:
            # Preprocess
            input_batch = self.transform(frame).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth_map = prediction.cpu().numpy()
            
            # Normalize for visualization
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"MiDaS depth estimation failed: {e}")
            return self._fallback_depth_estimation(frame)
    
    def _fallback_depth_estimation(self, frame: np.ndarray) -> np.ndarray:
        """Simple fallback depth estimation based on image intensity"""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame
            
            # Simple depth estimation based on intensity
            # Brighter areas are assumed to be closer
            depth_map = 1.0 - (gray / 255.0)
            
            # Apply Gaussian blur for smoothness
            from scipy.ndimage import gaussian_filter
            depth_map = gaussian_filter(depth_map, sigma=2.0)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Fallback depth estimation failed: {e}")
            return np.zeros_like(frame[:,:,0]) if len(frame.shape) == 3 else np.zeros_like(frame)
    
    def get_distance_estimation(self, frame: np.ndarray, bbox: tuple) -> float:
        """Estimate distance to object using depth map"""
        try:
            depth_map = self.estimate_depth(frame)
            
            # Extract depth in bounding box region
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                region_depth = depth_map[y1:y2, x1:x2]
                avg_depth = np.mean(region_depth)
                
                # Convert depth to distance (simplified)
                # Assuming depth map is normalized 0-1, convert to meters
                distance = 5.0 * (1.0 - avg_depth)  # 0-5 meters range
                return max(0.1, min(distance, 10.0))  # Clamp to reasonable range
            
            return 2.0  # Default distance
            
        except Exception as e:
            logger.error(f"Distance estimation failed: {e}")
            return 2.0  # Default distance 