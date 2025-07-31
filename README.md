# 🧭 SafeStep - Advanced Indoor Navigation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-Powered Indoor Navigation System for Visually Impaired and Accessibility**
>
> SafeStep is a comprehensive indoor navigation system that combines advanced computer vision, artificial intelligence, and multi-modal accessibility features to provide real-time navigation assistance in indoor environments.

## 🌟 Key Features

### 🤖 AI-Powered Detection

* **Real-time Object Detection**: YOLO-based detection of people, obstacles, doors, stairs, furniture
* **Depth Estimation**: MiDaS model for 3D depth mapping and distance calculation
* **Risk Assessment**: Intelligent risk level classification (high/medium/low/minimal)
* **Distance Estimation**: Accurate distance calculation to detected objects

### 🎤 Multi-Modal Accessibility

* **Voice Assistant**: Natural language processing for voice commands and feedback
* **Gesture Recognition**: Hand gesture control using MediaPipe
* **Text-to-Speech**: Real-time audio feedback for navigation instructions
* **Emergency Mode**: Quick activation for urgent situations

### 🧭 Advanced Navigation

* **Path Planning**: A\* algorithm with obstacle avoidance
* **Real-time Analytics**: Comprehensive detection logging and analytics
* **Safety Monitoring**: Continuous safety assessment and warnings
* **Multiple Modes**: Autonomous, Guided, Exploration, and Emergency modes

### 🖥️ Modern User Interface

* **Dark Theme**: Modern CustomTkinter interface
* **Real-time Visualization**: Live video feed with detection overlays
* **Analytics Dashboard**: Comprehensive statistics and data export
* **Responsive Design**: Adaptive layout for different screen sizes

## 📋 System Requirements

### Minimum Requirements

* **Python 3.8+** (3.9+ recommended)
* **4GB RAM** (8GB+ recommended)
* **Webcam** or USB camera
* **Microphone** (for voice commands)
* **Speakers/Headphones** (for voice feedback)

### Recommended

* **GPU with CUDA** (for faster AI processing)
* **8GB+ RAM** (for optimal performance)
* **High-resolution camera** (720p or higher)
* **Windows/Linux/macOS** (all supported)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/PathanWasim/SafeStep.git
cd SafeStep
```

### 2. Run Installation Script

```bash
python install.py
```

This automated script will:

* ✅ Check Python version compatibility
* ✅ Install PyTorch with appropriate CUDA support
* ✅ Install all required dependencies
* ✅ Set up system-specific dependencies
* ✅ Create configuration file
* ✅ Test all imports

### 3. Start the Application

```bash
# Run the modular version (recommended)
python main_new.py
# Or run the original version
python main.py
```

## 📁 Project Structure

```
SafeStep/
├── main_new.py              # 🚀 Main application (modular version)
├── main.py                  # Original monolithic version
├── install.py               # 🔧 Automated installation script
├── requirements.txt         # 📦 Python dependencies
├── config.json              # ⚙️ Configuration settings
├── README.md                # 📖 This documentation
├── README_MODULAR.md        # 📚 Detailed modular architecture
├── .gitignore               # 🚫 Git ignore rules
│
├── ai_models/               # 🤖 AI and ML components
│   ├── __init__.py
│   ├── depth_estimator.py   # MiDaS depth estimation
│   └── object_detector.py   # YOLO object detection
│
├── sensors/                 # 📡 Hardware interfaces
│   ├── __init__.py
│   ├── camera_manager.py    # Camera management
│   ├── gesture_controller.py# MediaPipe gestures
│   └── voice_assistant.py   # Speech recognition & TTS
│
├── navigation/              # 🧭 Navigation logic
│   ├── __init__.py
│   └── path_planner.py      # Path planning & obstacle avoidance
│
├── database/                # 💾 Data persistence
│   ├── __init__.py
│   └── database_manager.py  # SQLite database operations
│
├── models/                  # 📊 Data structures
│   ├── __init__.py
│   └── data_models.py       # Core data classes & enums
│
├── ui/                      # 🖥️ User interface
│   ├── __init__.py
│   └── main_window.py       # CustomTkinter UI components
│
└── Reports/                 # 📊 Research and documentation
    ├── Object Detection final.pptx
    └── Research Paper .pdf
```

## ⚙️ Configuration

The system uses `config.json` for configuration:

```json
{
  "camera": {
    "device_id": 0,
    "resolution": [1280, 720],
    "fps": 60
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
    "enabled": true,
    "language": "en-US",
    "rate": 150,
    "volume": 0.9
  },
  "ui": {
    "theme": "dark",
    "window_size": [1600, 1000],
    "fullscreen": false
  }
}
```

## 🎮 Usage Guide

### Voice Commands

* **"Navigate to bathroom"** - Find nearest restroom
* **"Find exit"** - Locate nearest exit
* **"Where am I?"** - Get current location
* **"What do you see?"** - Describe surroundings
* **"Emergency"** - Activate emergency mode

### Gesture Controls

* **🖐️ Open Palm** - Stop navigation
* **👆 Pointing** - Forward direction
* **👈 Left Hand** - Turn left
* **👉 Right Hand** - Turn right
* **✋ Raised Hand** - Help request

### UI Controls

* **Mode Selector** - Switch between navigation modes
* **Confidence Slider** - Adjust detection sensitivity
* **Voice Toggle** - Enable/disable voice assistant
* **Emergency Button** - Quick emergency activation
* **Export Data** - Export analytics and logs
  \`\`
