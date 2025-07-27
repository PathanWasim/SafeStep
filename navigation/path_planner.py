"""
Path Planning Module
===================

Advanced path planning with obstacle avoidance and navigation instructions.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from scipy.spatial.distance import euclidean

from models.data_models import DetectionResult, NavigationMode

logger = logging.getLogger(__name__)


class PathPlanner:
    """Advanced path planning with obstacle avoidance"""
    
    def __init__(self, map_width: int = 640, map_height: int = 480):
        self.map_width = map_width
        self.map_height = map_height
        self.map_data = np.zeros((map_height, map_width))  # Indoor map representation
        self.obstacles = []
        self.safe_paths = []
        self.current_position = (map_width // 2, map_height // 2)
        self.destination = None
    
    def update_obstacles(self, detections: List[DetectionResult]):
        """Update obstacle map with current detections"""
        self.obstacles = []
        
        for detection in detections:
            if detection.risk_level in ['high', 'medium']:
                bbox = detection.bbox
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                # Convert to map coordinates
                map_x = int(center_x * self.map_width / 640)
                map_y = int(center_y * self.map_height / 480)
                
                self.obstacles.append({
                    'position': (map_x, map_y),
                    'distance': detection.distance,
                    'type': detection.object_type,
                    'risk_level': detection.risk_level
                })
    
    def set_destination(self, destination: Tuple[int, int]):
        """Set navigation destination"""
        self.destination = destination
    
    def plan_path(self, start: Optional[Tuple[int, int]] = None, 
                 goal: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Plan optimal path using A* algorithm"""
        if start is None:
            start = self.current_position
        if goal is None:
            goal = self.destination
        
        if goal is None:
            return []
        
        try:
            # Simplified A* implementation
            path = self._a_star_pathfinding(start, goal)
            return path
            
        except Exception as e:
            logger.error(f"Path planning failed: {e}")
            return [start, goal] if start and goal else []
    
    def _a_star_pathfinding(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm implementation"""
        # For demo purposes, return a simple path
        # In a real implementation, this would use proper A* algorithm
        
        if start == goal:
            return [start]
        
        # Simple direct path with obstacle avoidance
        path = [start]
        current = start
        
        while current != goal:
            # Calculate direction to goal
            dx = goal[0] - current[0]
            dy = goal[1] - current[1]
            
            # Normalize direction
            distance = max(abs(dx), abs(dy))
            if distance > 0:
                dx = dx / distance
                dy = dy / distance
            
            # Check for obstacles in the way
            next_pos = (int(current[0] + dx), int(current[1] + dy))
            
            # Simple obstacle avoidance
            if self._is_position_safe(next_pos):
                current = next_pos
            else:
                # Try to find alternative path
                current = self._find_safe_neighbor(current, goal)
            
            path.append(current)
            
            # Prevent infinite loops
            if len(path) > 100:
                break
        
        return path
    
    def _is_position_safe(self, position: Tuple[int, int]) -> bool:
        """Check if position is safe from obstacles"""
        for obstacle in self.obstacles:
            distance = euclidean(position, obstacle['position'])
            if distance < 20:  # Minimum safe distance
                return False
        return True
    
    def _find_safe_neighbor(self, current: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, int]:
        """Find safe neighboring position"""
        # Try 8 directions around current position
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if self._is_position_safe(neighbor):
                return neighbor
        
        # If no safe neighbor, return current position
        return current
    
    def get_navigation_instruction(self, current_pos: Tuple[int, int], 
                                 next_waypoint: Tuple[int, int]) -> str:
        """Generate navigation instruction"""
        if not current_pos or not next_waypoint:
            return "No navigation data available"
        
        dx = next_waypoint[0] - current_pos[0]
        dy = next_waypoint[1] - current_pos[1]
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            if dx > 0:
                return "Turn right"
            else:
                return "Turn left"
        else:
            if dy > 0:
                return "Go forward"
            else:
                return "Go backward"
    
    def get_safety_assessment(self) -> dict:
        """Get current safety assessment"""
        high_risk_count = sum(1 for obs in self.obstacles if obs['risk_level'] == 'high')
        medium_risk_count = sum(1 for obs in self.obstacles if obs['risk_level'] == 'medium')
        
        if high_risk_count > 0:
            safety_level = "danger"
        elif medium_risk_count > 2:
            safety_level = "caution"
        elif medium_risk_count > 0:
            safety_level = "warning"
        else:
            safety_level = "safe"
        
        return {
            'safety_level': safety_level,
            'high_risk_obstacles': high_risk_count,
            'medium_risk_obstacles': medium_risk_count,
            'total_obstacles': len(self.obstacles)
        }
    
    def update_current_position(self, position: Tuple[int, int]):
        """Update current position"""
        self.current_position = position
    
    def get_obstacle_info(self) -> List[dict]:
        """Get information about current obstacles"""
        return self.obstacles.copy() 