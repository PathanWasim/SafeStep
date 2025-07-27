"""
Advanced Indoor Navigation System with AI-Powered Features
========================================================

A comprehensive indoor navigation solution combining:
- Real-time object detection and tracking
- Depth estimation and 3D mapping
- Voice-guided navigation with NLP
- Gesture recognition and control
- Multi-modal accessibility features
- Real-time analytics and insights
- Web-based dashboard integration
"""
import json
import numpy as np
import torch
import cv2
import torch.nn as nn
from ultralytics import YOLO
import mediapipe as mp
import speech_recognition as sr
import pyttsx3
import websockets
import requests
import customtkinter as ctk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import time
import logging
import sqlite3
import threading
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings("ignore")

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

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class NavigationMode(Enum):
    AUTONOMOUS = "autonomous"
    GUIDED = "guided"
    EXPLORATION = "exploration"
    EMERGENCY = "emergency"

class DetectionType(Enum):
    PERSON = "person"
    OBSTACLE = "obstacle"
    DOOR = "door"
    STAIRS = "stairs"
    CHAIR = "chair"
    TABLE = "table"

@dataclass
class DetectionResult:
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance: float
    depth_map: np.ndarray
    timestamp: datetime
    risk_level: str
    navigation_advice: str

@dataclass
class NavigationState:
    current_position: Tuple[float, float]
    destination: Optional[Tuple[float, float]]
    path: List[Tuple[float, float]]
    obstacles: List[DetectionResult]
    mode: NavigationMode
    is_safe: bool
    confidence: float

class DepthEstimator:
    """Advanced depth estimation using MiDaS model"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Load MiDaS model for depth estimation
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = self.transforms.small_transform
            
            self.enabled = True
            logger.info("Depth estimation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize depth estimation: {e}")
            self.enabled = False
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from RGB frame"""
        if not self.enabled:
            return np.zeros_like(frame[:,:,0])
        
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
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            return depth_map
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return np.zeros_like(frame[:,:,0])

class GestureController:
    """Hand gesture recognition for navigation control"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture commands mapping
        self.gestures = {
            'stop': self._detect_stop_gesture,
            'forward': self._detect_forward_gesture,
            'left': self._detect_left_gesture,
            'right': self._detect_right_gesture,
            'help': self._detect_help_gesture
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for gesture recognition"""
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        gesture_data = {
            'detected': False,
            'gesture': None,
            'confidence': 0.0,
            'landmarks': None
        }
        
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
        
        return gesture_data
    
    def _detect_stop_gesture(self, landmarks) -> float:
        """Detect stop gesture (open palm)"""
        # Implementation for stop gesture detection
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
    
    def _detect_forward_gesture(self, landmarks) -> float:
        """Detect forward gesture (pointing)"""
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
    
    def _detect_left_gesture(self, landmarks) -> float:
        """Detect left turn gesture"""
        palm_base = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        pinky_tip = landmarks.landmark[20]
        
        if thumb_tip.x < palm_base.x and pinky_tip.x < palm_base.x:
            return 0.85
        return 0.0
    
    def _detect_right_gesture(self, landmarks) -> float:
        """Detect right turn gesture"""
        palm_base = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        pinky_tip = landmarks.landmark[20]
        
        if thumb_tip.x > palm_base.x and pinky_tip.x > palm_base.x:
            return 0.85
        return 0.0
    
    def _detect_help_gesture(self, landmarks) -> float:
        """Detect help gesture (raised hand)"""
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        
        if middle_tip.y < wrist.y:
            return 0.92
        return 0.0

class VoiceAssistant:
    """Advanced voice assistant with NLP capabilities"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        
        # Voice commands mapping
        self.commands = {
            'navigate': self._handle_navigation_command,
            'where': self._handle_location_query,
            'what': self._handle_object_query,
            'help': self._handle_help_command,
            'stop': self._handle_stop_command,
            'emergency': self._handle_emergency_command
        }
        
        self.listening = False
        self.last_command_time = time.time()
    
    def start_listening(self):
        """Start continuous voice recognition"""
        self.listening = True
        thread = threading.Thread(target=self._listen_continuously)
        thread.daemon = True
        thread.start()
    
    def stop_listening(self):
        """Stop voice recognition"""
        self.listening = False
    
    def speak(self, text: str, priority: str = "normal"):
        """Text-to-speech with priority handling"""
        if priority == "emergency":
            # Clear queue for emergency messages
            self.tts_engine.stop()
        
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def _listen_continuously(self):
        """Continuous listening loop"""
        with self.microphone as source:
            # Calibrate once
            self.recognizer.adjust_for_ambient_noise(source)
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        logger.info(f"Voice command: {text}")
                        self._process_command(text)
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition error: {e}")
                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    logger.error(f"Voice assistant error: {e}")
                    time.sleep(1)
    
    def _process_command(self, text: str):
        """Process voice command"""
        for command, handler in self.commands.items():
            if command in text:
                handler(text)
                break
    
    def _handle_navigation_command(self, text: str):
        """Handle navigation commands"""
        if "bathroom" in text or "restroom" in text:
            self.speak("Navigating to nearest restroom")
        elif "exit" in text or "door" in text:
            self.speak("Searching for nearest exit")
        elif "elevator" in text:
            self.speak("Locating elevator")
        else:
            self.speak("Please specify your destination")
    
    def _handle_location_query(self, text: str):
        """Handle location queries"""
        self.speak("You are currently in the main corridor")
    
    def _handle_object_query(self, text: str):
        """Handle object identification queries"""
        self.speak("I can see several objects around you. Please be more specific")
    
    def _handle_help_command(self, text: str):
        """Handle help requests"""
        self.speak("I'm here to help you navigate. You can ask me to find exits, restrooms, or describe your surroundings")
    
    def _handle_stop_command(self, text: str):
        """Handle stop commands"""
        self.speak("Navigation stopped")
    
    def _handle_emergency_command(self, text: str):
        """Handle emergency commands"""
        self.speak("Emergency mode activated. Finding nearest exit", priority="emergency")

class DatabaseManager:
    """SQLite database for analytics and logging"""
    
    def __init__(self, db_path: str = "indoor_nav.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    distance REAL NOT NULL,
                    bbox TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    session_id TEXT NOT NULL
                )
            ''')
            
            # Navigation sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    mode TEXT NOT NULL,
                    total_detections INTEGER DEFAULT 0,
                    total_distance REAL DEFAULT 0.0
                )
            ''')
            
            # Analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    session_id TEXT
                )
            ''')
            
            conn.commit()
    
    def log_detection(self, detection: DetectionResult, session_id: str):
        """Log detection to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, object_type, confidence, distance, bbox, risk_level, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection.timestamp.isoformat(),
                detection.object_type,
                detection.confidence,
                detection.distance,
                json.dumps(detection.bbox),
                detection.risk_level,
                session_id
            ))
            conn.commit()
    
    def get_analytics_data(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get detection counts by type
            cursor.execute('''
                SELECT object_type, COUNT(*) as count
                FROM detections
                WHERE datetime(timestamp) >= datetime('now', '-{} days')
                GROUP BY object_type
                ORDER BY count DESC
            '''.format(days))
            
            detection_counts = dict(cursor.fetchall())
            
            # Get hourly activity
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM detections
                WHERE datetime(timestamp) >= datetime('now', '-{} days')
                GROUP BY hour
                ORDER BY hour
            '''.format(days))
            
            hourly_activity = dict(cursor.fetchall())
            
            return {
                'detection_counts': detection_counts,
                'hourly_activity': hourly_activity,
                'total_detections': sum(detection_counts.values()),
                'most_common_object': max(detection_counts.items(), key=lambda x: x[1])[0] if detection_counts else None
            }

class PathPlanner:
    """Advanced path planning with obstacle avoidance"""
    
    def __init__(self):
        self.map_data = np.zeros((480, 640))  # Indoor map representation
        self.obstacles = []
        self.safe_paths = []
    
    def update_obstacles(self, detections: List[DetectionResult]):
        """Update obstacle map with current detections"""
        self.obstacles = []
        
        for detection in detections:
            if detection.risk_level in ['high', 'medium']:
                bbox = detection.bbox
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                # Convert to map coordinates
                map_x = int(center_x * self.map_data.shape[1] / 640)
                map_y = int(center_y * self.map_data.shape[0] / 480)
                
                self.obstacles.append((map_x, map_y, detection.distance))
    
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan optimal path using A* algorithm"""
        # Simplified A* implementation
        path = []
        
        # For demo purposes, return a simple path
        if start and goal:
            path = [start, goal]
        
        return path
    
    def get_navigation_instruction(self, current_pos: Tuple[int, int], 
                                 next_waypoint: Tuple[int, int]) -> str:
        """Generate navigation instruction"""
        dx = next_waypoint[0] - current_pos[0]
        dy = next_waypoint[1] - current_pos[1]
        
        if abs(dx) > abs(dy):
            return "Turn right" if dx > 0 else "Turn left"
        else:
            return "Go forward" if dy > 0 else "Go backward"

class AdvancedIndoorNavigationSystem:
    """Main application class with advanced features"""
    
    def __init__(self):
        # Initialize core components
        self.setup_logging()
        self.load_config()
        self.init_ai_models()
        self.init_sensors()
        self.init_database()
        self.init_navigation()
        import uuid
        self.session_id = str(uuid.uuid4())
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
                "model_path": "yolov8n.pt"
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
    
    def init_ai_models(self):
        """Initialize AI models"""
        try:
            # YOLO for object detection
            model_path = self.config["detection"]["model_path"]
            self.yolo_model = YOLO(model_path)
            logger.info(f"YOLO model loaded: {model_path}")
            
            # Depth estimation
            self.depth_estimator = DepthEstimator()
            
            # Device info
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise
    
    def init_sensors(self):
        """Initialize sensors and input devices"""
        # Camera
        camera_id = self.config["camera"]["device_id"]
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            logger.warning(f"Failed to open camera {camera_id}")
            # Try default camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Failed to open any camera")
        
        # Set camera properties
        width, height = self.config["camera"]["resolution"]
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])
        
        # Gesture controller
        self.gesture_controller = GestureController()
        
        # Voice assistant
        if self.config["voice"]["enabled"]:
            self.voice_assistant = VoiceAssistant()
            self.voice_assistant.start_listening()
        
        logger.info("Sensors initialized successfully")
    
    def init_database(self):
        """Initialize database"""
        self.db_manager = DatabaseManager()
    
    def init_navigation(self):
        """Initialize navigation components"""
        self.path_planner = PathPlanner()
    
    def init_ui(self):
        """Initialize modern UI with CustomTkinter"""
        self.root = ctk.CTk()
        self.root.title("Advanced Indoor Navigation System")
        
        # Configure window
        width, height = self.config["ui"]["window_size"]
        self.root.geometry(f"{width}x{height}")
        
        if self.config["ui"]["fullscreen"]:
            self.root.attributes('-fullscreen', True)
        
        # Create UI layout
        self.create_modern_ui()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind('<Escape>', lambda e: self.toggle_fullscreen())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
    
    def create_modern_ui(self):
        """Create modern UI layout"""
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Top navigation bar
        self.create_navigation_bar()
        
        # Content area
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel (controls)
        self.create_control_panel()
        
        # Center panel (video and visualization)
        self.create_video_panel()
        
        # Right panel (analytics and info)
        self.create_analytics_panel()
        
        # Bottom status bar
        self.create_status_bar()
    
    def create_navigation_bar(self):
        """Create top navigation bar with modern design"""
        nav_frame = ctk.CTkFrame(self.main_frame, height=70, corner_radius=10)
        nav_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        nav_frame.grid_columnconfigure(0, weight=1)
        
        # Title with gradient
        title_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        title_frame.pack(side="left", padx=20)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🧭 NAVIGAIDE",
            font=ctk.CTkFont(size=28, weight="bold", family="Helvetica"),
            text_color="#4cc9f0"
        )
        title_label.pack(side="top")
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Advanced Indoor Navigation System",
            font=ctk.CTkFont(size=14, family="Helvetica"),
            text_color="#a0a0a0"
        )
        subtitle_label.pack(side="top", pady=(0, 5))
        
        # Right controls
        controls_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        controls_frame.pack(side="right", padx=20)
        
        # Navigation mode selector
        mode_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        mode_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(mode_frame, text="Mode:").pack(side="left", padx=(0, 5))
        self.mode_var = ctk.StringVar(value="Autonomous")
        mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            variable=self.mode_var,
            values=["Autonomous", "Guided", "Exploration", "Emergency"],
            command=self.on_mode_change,
            button_color="#4361ee",
            fg_color="#3a0ca3",
            dropdown_fg_color="#2b2d42"
        )
        mode_menu.pack(side="left")
        
        # Emergency button
        emergency_btn = ctk.CTkButton(
            controls_frame,
            text="🚨 EMERGENCY",
            command=self.activate_emergency_mode,
            fg_color="#d90429",
            hover_color="#ef233c",
            width=120,
            corner_radius=8,
            font=ctk.CTkFont(weight="bold")
        )
        emergency_btn.pack(side="left", padx=(10, 0))
    
    def create_control_panel(self):
        """Create left control panel with modern design"""
        control_frame = ctk.CTkFrame(self.content_frame, width=280, corner_radius=10)
        control_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        control_frame.grid_propagate(False)
        control_frame.grid_rowconfigure(4, weight=1)
        
        # Panel title
        panel_title = ctk.CTkLabel(
            control_frame, 
            text="CONTROL PANEL",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        panel_title.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")
        
        # Detection Settings
        detect_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        detect_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            detect_frame, 
            text="Detection Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Confidence threshold slider
        slider_frame = ctk.CTkFrame(detect_frame, fg_color="transparent")
        slider_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(slider_frame, text="Confidence:").pack(side="left")
        self.confidence_label = ctk.CTkLabel(slider_frame, text="0.5")
        self.confidence_label.pack(side="right")
        
        self.confidence_slider = ctk.CTkSlider(
            detect_frame, 
            from_=0.1, 
            to=1.0, 
            number_of_steps=18,
            command=self.on_confidence_change,
            button_color="#4361ee",
            button_hover_color="#3a0ca3"
        )
        self.confidence_slider.set(self.config["detection"]["confidence_threshold"])
        self.confidence_slider.pack(fill="x", padx=10, pady=(0, 10))
        
        # Voice Controls
        voice_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        voice_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            voice_frame, 
            text="Voice Assistant",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        self.voice_enabled = ctk.CTkSwitch(
            voice_frame, 
            text="Enable Voice Commands",
            command=self.toggle_voice_assistant,
            switch_width=50,
            switch_height=25,
            progress_color="#4361ee"
        )
        self.voice_enabled.pack(padx=10, pady=(5, 10))
        if self.config["voice"]["enabled"]:
            self.voice_enabled.select()
        
        # Quick Actions
        actions_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        actions_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            actions_frame, 
            text="Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        actions = [
            ("🚪 Find Exit", self.find_exit),
            ("🚻 Find Restroom", self.find_restroom),
            ("🛗 Find Elevator", self.find_elevator),
            ("📍 Current Location", self.announce_location),
            ("🔄 Recalibrate", self.recalibrate_system)
        ]
        
        for text, command in actions:
            btn = ctk.CTkButton(
                actions_frame, 
                text=text, 
                command=command,
                corner_radius=6,
                fg_color="#2b2d42",
                hover_color="#4a4e69",
                anchor="w"
            )
            btn.pack(fill="x", padx=10, pady=5)
        
        # System Status
        status_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        status_frame.grid(row=4, column=0, padx=10, pady=10, sticky="sew")
        
        ctk.CTkLabel(
            status_frame, 
            text="System Status",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        status_text = ctk.CTkTextbox(status_frame, height=80, activate_scrollbars=False)
        status_text.pack(fill="x", padx=10, pady=(0, 10))
        status_text.insert("1.0", "System operational\nAll sensors active")
        status_text.configure(state="disabled")
    
    def create_video_panel(self):
        """Create center video panel with modern design"""
        video_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        video_frame.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_rowconfigure(0, weight=1)
        
        # Video display container
        video_container = ctk.CTkFrame(video_frame)
        video_container.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Video display
        self.video_label = ctk.CTkLabel(video_container, text="", corner_radius=8)
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Overlay information
        overlay_frame = ctk.CTkFrame(video_frame, height=120, corner_radius=8)
        overlay_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Current instruction
        self.instruction_label = ctk.CTkLabel(
            overlay_frame,
            text="System Ready - Voice commands active",
            font=ctk.CTkFont(size=18, weight="bold"),
            wraplength=600
        )
        self.instruction_label.pack(pady=10, padx=20, anchor="w")
        
        # Navigation info
        nav_info_frame = ctk.CTkFrame(overlay_frame, fg_color="transparent")
        nav_info_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.distance_label = ctk.CTkLabel(
            nav_info_frame, 
            text="Distance: --",
            font=ctk.CTkFont(size=14)
        )
        self.distance_label.pack(side="left", padx=10)
        
        self.direction_label = ctk.CTkLabel(
            nav_info_frame, 
            text="Direction: --",
            font=ctk.CTkFont(size=14)
        )
        self.direction_label.pack(side="left", padx=10)
        
        self.safety_label = ctk.CTkLabel(
            nav_info_frame, 
            text="✅ Safe",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.safety_label.pack(side="right", padx=10)
    
    def create_analytics_panel(self):
        """Create right analytics panel with modern design"""
        analytics_frame = ctk.CTkFrame(self.content_frame, width=350, corner_radius=10)
        analytics_frame.grid(row=0, column=2, padx=(10, 0), pady=0, sticky="nsew")
        analytics_frame.grid_propagate(False)
        analytics_frame.grid_rowconfigure(1, weight=1)
        
        # Panel title
        panel_title = ctk.CTkLabel(
            analytics_frame, 
            text="ANALYTICS DASHBOARD",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        panel_title.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")
        
        # Tab view for different analytics
        self.analytics_tabs = ctk.CTkTabview(analytics_frame)
        self.analytics_tabs.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        # Real-time stats tab
        realtime_tab = self.analytics_tabs.add("Realtime")
        realtime_tab.grid_columnconfigure(0, weight=1)
        
        stats_frame = ctk.CTkScrollableFrame(realtime_tab)
        stats_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.stats_text = ctk.CTkTextbox(stats_frame, height=300)
        self.stats_text.pack(fill="both", expand=True)
        
        # History tab
        history_tab = self.analytics_tabs.add("History")
        history_tab.grid_columnconfigure(0, weight=1)
        
        history_frame = ctk.CTkScrollableFrame(history_tab)
        history_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.history_text = ctk.CTkTextbox(history_frame)
        self.history_text.pack(fill="both", expand=True)
        
        # Actions at the bottom
        btn_frame = ctk.CTkFrame(analytics_frame)
        btn_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        export_btn = ctk.CTkButton(
            btn_frame, 
            text="📊 Export Data", 
            command=self.export_analytics,
            corner_radius=6,
            fg_color="#2b2d42",
            hover_color="#4a4e69"
        )
        export_btn.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        clear_btn = ctk.CTkButton(
            btn_frame, 
            text="🗑️ Clear History", 
            command=self.clear_history,
            corner_radius=6,
            fg_color="#2b2d42",
            hover_color="#4a4e69"
        )
        clear_btn.pack(side="right", padx=5, pady=5, fill="x", expand=True)
    
    def create_status_bar(self):
        """Create bottom status bar with modern design"""
        status_frame = ctk.CTkFrame(self.main_frame, height=40, corner_radius=10)
        status_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        # Status indicators
        self.fps_label = ctk.CTkLabel(
            status_frame, 
            text="FPS: 0",
            font=ctk.CTkFont(size=12),
            padx=10
        )
        self.fps_label.pack(side="left")
        
        self.detection_count_label = ctk.CTkLabel(
            status_frame, 
            text="Detections: 0",
            font=ctk.CTkFont(size=12),
            padx=10
        )
        self.detection_count_label.pack(side="left")
        
        # Session info
        session_label = ctk.CTkLabel(
            status_frame, 
            text=f"Session: {self.session_id}",
            font=ctk.CTkFont(size=12),
            padx=10
        )
        session_label.pack(side="left")
        
        self.system_status_label = ctk.CTkLabel(
            status_frame, 
            text="🟢 System Online",
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=10
        )
        self.system_status_label.pack(side="right")
    
    def process_frame(self):
        """Main frame processing loop"""
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            # Show placeholder when camera fails
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "CAMERA NOT AVAILABLE", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.update_video_display(placeholder)
            return
        
        original_frame = frame.copy()
        
        try:
            # YOLO detection
            results = self.yolo_model.predict(
                frame, 
                conf=self.config["detection"]["confidence_threshold"],
                verbose=False
            )
            
            # Process detections
            detections = []
            detection_count = 0
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    detection = self.process_detection(box, results[0].names, frame)
                    if detection:
                        detections.append(detection)
                        detection_count += 1
                        
                        # Draw detection on frame
                        self.draw_detection(frame, detection)
            
            # Gesture recognition
            gesture_data = self.gesture_controller.process_frame(frame)
            if gesture_data['detected']:
                self.handle_gesture_command(gesture_data)
            
            # Update navigation state
            self.update_navigation_state(detections)
            
            # Update UI
            self.update_video_display(frame)
            self.update_real_time_stats(detection_count, detections)
            
            # Calculate FPS
            self.update_fps()
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def process_detection(self, box, names, frame) -> Optional[DetectionResult]:
        """Process individual detection"""
        try:
            # Extract detection data
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            object_type = names[class_id]
            
            # Calculate distance (simplified estimation)
            object_width = bbox[2] - bbox[0]
            object_height = bbox[3] - bbox[1]
            
            # Rough distance estimation based on object size
            known_widths = {
                'person': 0.5,  # Average shoulder width in meters
                'chair': 0.5,
                'table': 1.0,
                'door': 0.9,
                'car': 1.8
            }
            
            if object_type in known_widths:
                focal_length = frame.shape[1] / (2 * np.tan(np.radians(35)))
                distance = (known_widths[object_type] * focal_length) / (object_width + 1e-6)
                distance = max(0.1, min(distance, 10.0))  # Clamp distance
            else:
                distance = 2.0  # Default distance
            
            # Determine risk level
            risk_level = self.assess_risk_level(object_type, distance, bbox, frame.shape)
            
            # Generate navigation advice
            navigation_advice = self.generate_navigation_advice(object_type, distance, risk_level)
            
            # Create detection result
            detection = DetectionResult(
                object_type=object_type,
                confidence=confidence,
                bbox=tuple(bbox),
                distance=distance,
                depth_map=np.array([]),  # Placeholder
                timestamp=datetime.now(),
                risk_level=risk_level,
                navigation_advice=navigation_advice
            )
            
            # Log to database
            self.db_manager.log_detection(detection, self.session_id)
            
            return detection
            
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            return None
    
    def assess_risk_level(self, object_type: str, distance: float, 
                         bbox: np.ndarray, frame_shape: tuple) -> str:
        """Assess risk level based on object type and distance"""
        # Distance-based risk
        if distance < self.config["navigation"]["emergency_distance"]:
            return "high"
        elif distance < self.config["navigation"]["warning_distance"]:
            return "medium"
        elif distance < self.config["navigation"]["safe_distance"]:
            return "low"
        else:
            return "minimal"
    
    def generate_navigation_advice(self, object_type: str, distance: float, 
                                 risk_level: str) -> str:
        """Generate navigation advice based on detection"""
        if risk_level == "high":
            return f"STOP! {object_type.title()} directly ahead at {distance:.1f}m"
        elif risk_level == "medium":
            return f"Caution: {object_type} at {distance:.1f}m - prepare to navigate around"
        elif risk_level == "low":
            return f"{object_type.title()} detected at {distance:.1f}m"
        else:
            return f"{object_type.title()} in view"
    
    def draw_detection(self, frame: np.ndarray, detection: DetectionResult):
        """Draw detection visualization on frame"""
        bbox = detection.bbox
        
        # Color based on risk level
        colors = {
            "high": (0, 0, 255),      # Red
            "medium": (0, 165, 255),  # Orange
            "low": (0, 255, 255),     # Yellow
            "minimal": (0, 255, 0)    # Green
        }
        
        color = colors.get(detection.risk_level, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label
        label = f"{detection.object_type} ({detection.distance:.1f}m) {detection.confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for label
        cv2.rectangle(frame, 
                     (bbox[0], bbox[1] - label_size[1] - 10),
                     (bbox[0] + label_size[0], bbox[1]),
                     color, -1)
        
        # Label text
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Risk indicator
        if detection.risk_level in ["high", "medium"]:
            cv2.circle(frame, (bbox[0] + 20, bbox[1] + 20), 10, color, -1)
    
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
        if high_risk_objects and hasattr(self, 'voice_assistant'):
            for obj in high_risk_objects:
                if time.time() - getattr(obj, 'last_announced', 0) > 3:  # Cooldown
                    self.voice_assistant.speak(obj.navigation_advice, priority="high")
                    obj.last_announced = time.time()
        
        # Update path planner
        self.path_planner.update_obstacles(detections)
    
    def update_video_display(self, frame: np.ndarray):
        """Update video display in UI"""
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit display
            display_height = 480
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            
            frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_resized)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep reference
            
        except Exception as e:
            logger.error(f"Error updating video display: {e}")
    
    def update_real_time_stats(self, detection_count: int, detections: List[DetectionResult]):
        """Update real-time statistics display"""
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
            
            # Update stats display
            stats_text = f"""
Current Frame Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━
Total Objects: {detection_count}
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
            
            # Update UI
            self.stats_text.configure(state="normal")
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("1.0", stats_text)
            self.stats_text.configure(state="disabled")
            
            # Update navigation info
            if closest_object:
                self.distance_label.configure(text=f"Distance: {closest_object.distance:.1f}m")
                self.direction_label.configure(text=f"Object: {closest_object.object_type.title()}")
                
                if closest_object.risk_level == "high":
                    self.safety_label.configure(text="🔴 DANGER", text_color="#d90429")
                elif closest_object.risk_level == "medium":
                    self.safety_label.configure(text="🟡 CAUTION", text_color="#ff9e00")
                else:
                    self.safety_label.configure(text="✅ Safe", text_color="#2ec4b6")
            
            # Update instruction
            if detections:
                high_risk = [d for d in detections if d.risk_level == "high"]
                if high_risk:
                    self.instruction_label.configure(
                        text=f"⚠️ {high_risk[0].navigation_advice}",
                        text_color="#d90429"
                    )
                else:
                    self.instruction_label.configure(
                        text="Continue forward - Path clear",
                        text_color="#2ec4b6"
                    )
            else:
                self.instruction_label.configure(
                    text="No obstacles detected - Safe to proceed",
                    text_color="#2ec4b6"
                )
            
        except Exception as e:
            logger.error(f"Error updating real-time stats: {e}")
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
            
            # Update UI
            self.fps_label.configure(text=f"FPS: {self.current_fps}")
    
    def handle_gesture_command(self, gesture_data: Dict[str, Any]):
        """Handle gesture-based commands"""
        gesture = gesture_data['gesture']
        confidence = gesture_data['confidence']
        
        logger.info(f"Gesture detected: {gesture} (confidence: {confidence:.2f})")
        
        if gesture == 'stop':
            self.voice_assistant.speak("Navigation stopped by gesture")
        elif gesture == 'help':
            self.voice_assistant.speak("Gesture help activated")
        # Add more gesture handlers as needed
    
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
        
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.speak(f"Navigation mode changed to {mode}")
    
    def on_confidence_change(self, value):
        """Handle confidence threshold change"""
        self.config["detection"]["confidence_threshold"] = value
        self.confidence_label.configure(text=f"{value:.2f}")
    
    def toggle_voice_assistant(self):
        """Toggle voice assistant on/off"""
        if hasattr(self, 'voice_assistant'):
            if self.voice_enabled.get():
                self.voice_assistant.start_listening()
                logger.info("Voice assistant enabled")
            else:
                self.voice_assistant.stop_listening()
                logger.info("Voice assistant disabled")
    
    def activate_emergency_mode(self):
        """Activate emergency mode"""
        self.current_state.mode = NavigationMode.EMERGENCY
        self.mode_var.set("Emergency")
        
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.speak("Emergency mode activated. Finding nearest exit.", priority="emergency")
        
        logger.warning("Emergency mode activated")
        
        # Change UI to emergency theme
        self.root.configure(fg_color="#2b0706")
    
    def find_exit(self):
        """Find nearest exit"""
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.speak("Searching for nearest exit")
        logger.info("Finding exit requested")
    
    def find_restroom(self):
        """Find nearest restroom"""
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.speak("Searching for nearest restroom")
        logger.info("Finding restroom requested")
    
    def find_elevator(self):
        """Find nearest elevator"""
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.speak("Searching for elevator")
        logger.info("Finding elevator requested")
    
    def announce_location(self):
        """Announce current location"""
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.speak("You are currently in the main corridor")
        logger.info("Location announcement requested")
    
    def recalibrate_system(self):
        """Recalibrate the system"""
        if hasattr(self, 'voice_assistant'):
            self.voice_assistant.speak("System recalibration initiated")
        logger.info("System recalibration requested")
    
    def export_analytics(self):
        """Export analytics data"""
        try:
            analytics_data = self.db_manager.get_analytics_data()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_export_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(analytics_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", f"Analytics exported to {filename}")
            logger.info(f"Analytics exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
            messagebox.showerror("Export Error", str(e))
    
    def clear_history(self):
        """Clear detection history"""
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.configure(state="disabled")
        logger.info("Detection history cleared")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)
    
    def on_closing(self):
        """Handle application closing"""
        try:
            logger.info("Shutting down system...")
            
            # Stop voice assistant
            if hasattr(self, 'voice_assistant'):
                self.voice_assistant.stop_listening()
            
            # Release camera
            if hasattr(self, 'camera'):
                self.camera.release()
            
            # Save configuration
            self.save_config()
            
            self.root.destroy()
            sys.exit(0)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.root.destroy()
            sys.exit(1)
    
    def run(self):
        """Start the main application"""
        try:
            logger.info("Starting Advanced Indoor Navigation System")
            self.is_running = True
            
            # Start main processing loop
            self.process_loop()
            
            # Start UI main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error running application: {e}")
            raise
    
    def process_loop(self):
        """Main processing loop"""
        if self.is_running:
            try:
                self.process_frame()
                
                # Update detection count
                detection_count = len(self.current_state.obstacles)
                self.detection_count_label.configure(text=f"Detections: {detection_count}")
                
                # Schedule next iteration
                self.root.after(33, self.process_loop)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.system_status_label.configure(text="🔴 System Error", text_color="#d90429")
                self.root.after(1000, self.process_loop)  # Retry after 1 second


def main():
    """Main entry point"""
    try:
        # Check system requirements
        print("🔍 Checking system requirements...")
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        print("✅ Python version OK")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - using CPU")
        
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