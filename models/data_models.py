"""
Data Models for Indoor Navigation System
=======================================

Core data structures, enums, and dataclasses used throughout the system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class NavigationMode(Enum):
    """Navigation modes for the system"""
    AUTONOMOUS = "autonomous"
    GUIDED = "guided"
    EXPLORATION = "exploration"
    EMERGENCY = "emergency"


class DetectionType(Enum):
    """Types of objects that can be detected"""
    PERSON = "person"
    OBSTACLE = "obstacle"
    DOOR = "door"
    STAIRS = "stairs"
    CHAIR = "chair"
    TABLE = "table"


@dataclass
class DetectionResult:
    """Result of object detection with metadata"""
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance: float
    depth_map: np.ndarray
    timestamp: datetime
    risk_level: str
    navigation_advice: str
    last_announced: float = field(default=0.0, init=False)
    
    def __post_init__(self):
        """Initialize additional attributes after dataclass creation"""
        if not hasattr(self, 'last_announced'):
            self.last_announced = 0.0


@dataclass
class NavigationState:
    """Current state of navigation system"""
    current_position: Tuple[float, float]
    destination: Optional[Tuple[float, float]]
    path: List[Tuple[float, float]]
    obstacles: List[DetectionResult]
    mode: NavigationMode
    is_safe: bool
    confidence: float


@dataclass
class GestureData:
    """Gesture recognition data"""
    detected: bool = False
    gesture: Optional[str] = None
    confidence: float = 0.0
    landmarks: Optional[Any] = None


@dataclass
class SystemConfig:
    """System configuration data"""
    camera: Dict[str, Any]
    detection: Dict[str, Any]
    navigation: Dict[str, Any]
    voice: Dict[str, Any]
    ui: Dict[str, Any] 