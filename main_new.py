"""
Advanced Indoor Navigation System - Modular Version
==================================================

A comprehensive indoor navigation solution with modular architecture.
"""

import json
import numpy as np
import cv2
import time
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# Import modular components
from models.data_models import NavigationMode, DetectionResult, NavigationState
from ai_models.depth_estimator import DepthEstimator
from ai_models.object_detector import ObjectDetector
from sensors.camera_manager import CameraManager
from sensors.gesture_controller import GestureController
from sensors.voice_assistant import VoiceAssistant
from database.database_manager import DatabaseManager
from navigation.path_planner import PathPlanner
from ui.main_window import MainWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indoor_nav.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedIndoorNavigationSystem:
    """Main application class with modular architecture"""
    
    def __init__(self):
        # Initialize core components
        self.setup_logging()
        self.load_config()
        self.init_components()
        self.init_ui()
        
        # Application state
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_running = False
        self.current_state = NavigationState(
            current_position=(0, 0),
            destination=None,
            path=[],
            obstacles=[],
            mode=NavigationMode.AUTONOMOUS,
            is_safe=True,
            confidence=0.0
        )
        
        # Performance metrics
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        logger.info("Advanced Indoor Navigation System initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Already configured globally
        pass
    
    def load_config(self):
        """Load configuration from file"""
        config_file = Path("config.json")
        default_config = {
            "camera": {
                "device_id": 0,
                "resolution": [640, 480],
                "fps": 30
            },
            "detection": {
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "model_path": "yolo11n.pt"
            },
            "navigation": {
                "safe_distance": 2.0,
                "warning_distance": 1.0,
                "emergency_distance": 0.5
            },
            "voice": {
                "enabled": True,
                "language": "en-US",
                "rate": 150,
                "volume": 0.9
            },
            "ui": {
                "theme": "dark",
                "window_size": [1400, 900],
                "fullscreen": False
            }
        }
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open("config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def init_components(self):
        """Initialize all system components"""
        try:
            # Initialize AI models
            self.object_detector = ObjectDetector(
                model_path=self.config["detection"]["model_path"],
                confidence_threshold=self.config["detection"]["confidence_threshold"]
            )
            
            self.depth_estimator = DepthEstimator(enable_midas=True)
            
            # Initialize sensors
            camera_config = self.config["camera"]
            self.camera_manager = CameraManager(
                device_id=camera_config["device_id"],
                resolution=tuple(camera_config["resolution"]),
                fps=camera_config["fps"]
            )
            
            self.gesture_controller = GestureController(enable_mediapipe=True)
            
            if self.config["voice"]["enabled"]:
                self.voice_assistant = VoiceAssistant(enable_speech=True, enable_tts=True)
                self.voice_assistant.start_listening()
            else:
                self.voice_assistant = None
            
            # Initialize database
            self.db_manager = DatabaseManager()
            
            # Initialize navigation
            self.path_planner = PathPlanner()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def init_ui(self):
        """Initialize user interface"""
        try:
            ui_config = self.config["ui"]
            self.main_window = MainWindow(
                title="SafeStep - Indoor Navigation System",
                width=ui_config["window_size"][0],
                height=ui_config["window_size"][1]
            )
            
            # Set up UI callbacks
            callbacks = {
                'mode_change': self.on_mode_change,
                'confidence_change': self.on_confidence_change,
                'voice_toggle': self.on_voice_toggle,
                'emergency': self.on_emergency,
                'export': self.on_export,
                'clear': self.on_clear,
                'closing': self.on_closing
            }
            self.main_window.set_callbacks(callbacks)
            
            logger.info("UI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize UI: {e}")
            raise
    
    def process_frame(self):
        """Main frame processing loop"""
        # Read frame from camera
        ret, frame = self.camera_manager.read_frame()
        if not ret or frame is None:
            logger.warning("Failed to read frame from camera")
            frame = self.camera_manager.get_placeholder_frame()
        
        if frame is None:
            return
        
        original_frame = frame.copy()
        
        try:
            # Object detection
            detections = self.object_detector.detect_objects(frame)
            
            # Draw detections on frame
            frame = self.object_detector.draw_detections(frame, detections)
            
            # Gesture recognition
            gesture_data = self.gesture_controller.process_frame(frame)
            if gesture_data['detected']:
                self.handle_gesture_command(gesture_data)
            
            # Update navigation state
            self.update_navigation_state(detections)
            
            # Update path planner
            self.path_planner.update_obstacles(detections)
            
            # Update UI
            self.update_ui(frame, detections)
            
            # Calculate FPS
            self.update_fps()
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def update_navigation_state(self, detections: List[DetectionResult]):
        """Update navigation state based on current detections"""
        # Update obstacles
        self.current_state.obstacles = detections
        
        # Assess overall safety
        high_risk_objects = [d for d in detections if d.risk_level == "high"]
        medium_risk_objects = [d for d in detections if d.risk_level == "medium"]
        
        self.current_state.is_safe = len(high_risk_objects) == 0
        
        # Calculate confidence
        if detections:
            avg_confidence = sum(d.confidence for d in detections) / len(detections)
            self.current_state.confidence = avg_confidence
        
        # Generate voice feedback for high-risk situations
        if high_risk_objects and self.voice_assistant:
            for obj in high_risk_objects:
                if time.time() - obj.last_announced > 3:  # Cooldown
                    self.voice_assistant.speak(obj.navigation_advice, priority="high")
                    obj.last_announced = time.time()
    
    def update_ui(self, frame: np.ndarray, detections: List[DetectionResult]):
        """Update user interface with current data"""
        try:
            # Update video display
            self.main_window.update_video_frame(frame)
            
            # Update statistics
            stats_text = self.generate_stats_text(detections)
            self.main_window.update_stats(stats_text)
            
            # Update navigation info
            if detections:
                closest_detection = min(detections, key=lambda d: d.distance)
                self.main_window.update_instruction(
                    closest_detection.navigation_advice,
                    "#d90429" if closest_detection.risk_level == "high" else "#2ec4b6"
                )
            else:
                self.main_window.update_instruction(
                    "No obstacles detected - Safe to proceed",
                    "#2ec4b6"
                )
            
            # Update detection count
            self.main_window.update_detection_count(len(detections))
            
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
    
    def generate_stats_text(self, detections: List[DetectionResult]) -> str:
        """Generate statistics text for display"""
        try:
            # Calculate statistics
            object_types = {}
            total_distance = 0
            closest_object = None
            min_distance = float('inf')
            
            for detection in detections:
                # Count by type
                obj_type = detection.object_type
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
                
                # Track closest object
                if detection.distance < min_distance:
                    min_distance = detection.distance
                    closest_object = detection
            
            # Generate stats text
            stats_text = f"""
Current Frame Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━
Total Objects: {len(detections)}
Unique Types: {len(object_types)}

Object Breakdown:
"""
            
            for obj_type, count in sorted(object_types.items()):
                stats_text += f"  • {obj_type.title()}: {count}\n"
            
            if closest_object:
                stats_text += f"""
Closest Object:
  • Type: {closest_object.object_type.title()}
  • Distance: {closest_object.distance:.1f}m
  • Risk Level: {closest_object.risk_level.title()}
  • Confidence: {closest_object.confidence:.2f}
"""
            
            # Safety status
            safety_status = "🟢 SAFE" if self.current_state.is_safe else "🔴 CAUTION"
            stats_text += f"\nSafety Status: {safety_status}"
            
            return stats_text
            
        except Exception as e:
            logger.error(f"Error generating stats text: {e}")
            return "Error generating statistics"
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
            
            # Update UI
            self.main_window.update_fps(self.current_fps)
    
    def handle_gesture_command(self, gesture_data: Dict):
        """Handle gesture-based commands"""
        gesture = gesture_data['gesture']
        confidence = gesture_data['confidence']
        
        logger.info(f"Gesture detected: {gesture} (confidence: {confidence:.2f})")
        
        if self.voice_assistant:
            if gesture == 'stop':
                self.voice_assistant.speak("Navigation stopped by gesture")
            elif gesture == 'help':
                self.voice_assistant.speak("Gesture help activated")
    
    # Event handlers
    def on_mode_change(self, mode: str):
        """Handle navigation mode change"""
        mode_mapping = {
            "Autonomous": NavigationMode.AUTONOMOUS,
            "Guided": NavigationMode.GUIDED,
            "Exploration": NavigationMode.EXPLORATION,
            "Emergency": NavigationMode.EMERGENCY
        }
        
        self.current_state.mode = mode_mapping[mode]
        logger.info(f"Navigation mode changed to: {mode}")
        
        if self.voice_assistant:
            self.voice_assistant.speak(f"Navigation mode changed to {mode}")
    
    def on_confidence_change(self, value):
        """Handle confidence threshold change"""
        self.config["detection"]["confidence_threshold"] = value
        self.object_detector.update_confidence_threshold(value)
    
    def on_voice_toggle(self, enabled: bool):
        """Handle voice assistant toggle"""
        if self.voice_assistant:
            if enabled:
                self.voice_assistant.start_listening()
                logger.info("Voice assistant enabled")
            else:
                self.voice_assistant.stop_listening()
                logger.info("Voice assistant disabled")
    
    def on_emergency(self):
        """Handle emergency mode activation"""
        self.current_state.mode = NavigationMode.EMERGENCY
        
        if self.voice_assistant:
            self.voice_assistant.speak("Emergency mode activated. Finding nearest exit.", priority="emergency")
        
        logger.warning("Emergency mode activated")
    
    def on_export(self):
        """Handle data export"""
        try:
            filename = self.db_manager.export_data()
            if filename:
                logger.info(f"Data exported to {filename}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    def on_clear(self):
        """Handle history clear"""
        try:
            self.db_manager.clear_old_data(days=0)  # Clear all data
            logger.info("History cleared")
        except Exception as e:
            logger.error(f"Clear failed: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        try:
            logger.info("Shutting down system...")
            
            # Stop voice assistant
            if self.voice_assistant:
                self.voice_assistant.stop_listening()
            
            # Release camera
            if self.camera_manager:
                self.camera_manager.release()
            
            # Save configuration
            self.save_config()
            
            # End session
            self.db_manager.end_session(self.session_id, len(self.current_state.obstacles))
            
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def run(self):
        """Start the main application"""
        try:
            logger.info("Starting Advanced Indoor Navigation System")
            self.is_running = True
            
            # Log session start
            self.db_manager.log_session(
                self.session_id, 
                self.current_state.mode.value, 
                datetime.now()
            )
            
            # Start main processing loop
            self.process_loop()
            
            # Start UI main loop
            self.main_window.run()
            
        except Exception as e:
            logger.error(f"Error running application: {e}")
            raise
    
    def process_loop(self):
        """Main processing loop optimized for 60 FPS"""
        if self.is_running:
            try:
                start_time = time.time()
                
                self.process_frame()
                
                # Calculate processing time and adjust delay for 60 FPS
                processing_time = time.time() - start_time
                target_frame_time = 1.0 / 60.0  # 60 FPS = 16.67ms per frame
                delay = max(1, int((target_frame_time - processing_time) * 1000))
                
                # Schedule next iteration
                self.main_window.root.after(delay, self.process_loop)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.main_window.update_system_status("🔴 System Error", "#d90429")
                self.main_window.root.after(1000, self.process_loop)  # Retry after 1 second


def main():
    """Main entry point"""
    try:
        # Check system requirements
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        print("✅ Python version OK")
        
        # Check PyTorch availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  CUDA not available - using CPU")
        except ImportError:
            print("⚠️  PyTorch not available - some features may not work")
        
        print("\n🚀 Launching Advanced Indoor Navigation System...")
        
        # Create and run application
        app = AdvancedIndoorNavigationSystem()
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 