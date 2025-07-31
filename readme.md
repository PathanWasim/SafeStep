# 🧭 SafeStep - Advanced Indoor Navigation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-Powered Indoor Navigation System for Visually Impaired and Accessibility**

SafeStep is a comprehensive indoor navigation system that combines advanced computer vision, artificial intelligence, and multi-modal accessibility features to provide real-time navigation assistance in indoor environments.

## 🌟 Key Features

### 🤖 AI-Powered Detection
- **Real-time Object Detection**: YOLO-based detection of people, obstacles, doors, stairs, furniture
- **Depth Estimation**: MiDaS model for 3D depth mapping and distance calculation
- **Risk Assessment**: Intelligent risk level classification (high/medium/low/minimal)
- **Distance Estimation**: Accurate distance calculation to detected objects

### 🎤 Multi-Modal Accessibility
- **Voice Assistant**: Natural language processing for voice commands and feedback
- **Gesture Recognition**: Hand gesture control using MediaPipe
- **Text-to-Speech**: Real-time audio feedback for navigation instructions
- **Emergency Mode**: Quick activation for urgent situations

### 🧭 Advanced Navigation
- **Path Planning**: A* algorithm with obstacle avoidance
- **Real-time Analytics**: Comprehensive detection logging and analytics
- **Safety Monitoring**: Continuous safety assessment and warnings
- **Multiple Modes**: Autonomous, Guided, Exploration, and Emergency modes

### 🖥️ Modern User Interface
- **Dark Theme**: Modern CustomTkinter interface
- **Real-time Visualization**: Live video feed with detection overlays
- **Analytics Dashboard**: Comprehensive statistics and data export
- **Responsive Design**: Adaptive layout for different screen sizes

## 📋 System Requirements

### Minimum Requirements
- **Python 3.8+** (3.9+ recommended)
- **4GB RAM** (8GB+ recommended)
- **Webcam** or USB camera
- **Microphone** (for voice commands)
- **Speakers/Headphones** (for voice feedback)

### Recommended
- **GPU with CUDA** (for faster AI processing)
- **8GB+ RAM** (for optimal performance)
- **High-resolution camera** (720p or higher)
- **Windows/Linux/macOS** (all supported)

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
- ✅ Check Python version compatibility
- ✅ Install PyTorch with appropriate CUDA support
- ✅ Install all required dependencies
- ✅ Set up system-specific dependencies
- ✅ Create configuration file
- ✅ Test all imports

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
├── config.json             # ⚙️ Configuration settings
├── README.md               # 📖 This documentation
├── README_MODULAR.md       # 📚 Detailed modular architecture
├── .gitignore              # 🚫 Git ignore rules
│
├── ai_models/              # 🤖 AI and ML components
│   ├── __init__.py
│   ├── depth_estimator.py  # MiDaS depth estimation
│   └── object_detector.py  # YOLO object detection
│
├── sensors/                # 📡 Hardware interfaces
│   ├── __init__.py
│   ├── camera_manager.py   # Camera management
│   ├── gesture_controller.py # MediaPipe gestures
│   └── voice_assistant.py  # Speech recognition & TTS
│
├── navigation/             # 🧭 Navigation logic
│   ├── __init__.py
│   └── path_planner.py     # Path planning & obstacle avoidance
│
├── database/               # 💾 Data persistence
│   ├── __init__.py
│   └── database_manager.py # SQLite database operations
│
├── models/                 # 📊 Data structures
│   ├── __init__.py
│   └── data_models.py      # Core data classes & enums
│
├── ui/                     # 🖥️ User interface
│   ├── __init__.py
│   └── main_window.py      # CustomTkinter UI components
│
└── Reports/                # 📊 Research and documentation
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
- **"Navigate to bathroom"** - Find nearest restroom
- **"Find exit"** - Locate nearest exit
- **"Where am I?"** - Get current location
- **"What do you see?"** - Describe surroundings
- **"Emergency"** - Activate emergency mode

### Gesture Controls
- **🖐️ Open Palm** - Stop navigation
- **👆 Pointing** - Forward direction
- **👈 Left Hand** - Turn left
- **👉 Right Hand** - Turn right
- **✋ Raised Hand** - Help request

### UI Controls
- **Mode Selector** - Switch between navigation modes
- **Confidence Slider** - Adjust detection sensitivity
- **Voice Toggle** - Enable/disable voice assistant
- **Emergency Button** - Quick emergency activation
- **Export Data** - Export analytics and logs

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Run installation script
python install.py

# Or manually install dependencies
pip install -r requirements.txt
```

#### Camera Not Working
```bash
# Check camera availability
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"

# Update config.json camera.device_id if needed
```

#### Voice Features Not Working
```bash
# Test microphone
python -c "import speech_recognition as sr; print('Microphones:', sr.Microphone.list_microphone_names())"

# Test TTS
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait()"
```

#### GPU/CUDA Issues
```bash
# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Install CPU version if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Performance Optimization

#### For Better Performance
1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Reduce Resolution**: Lower camera resolution in config
3. **Adjust Confidence**: Increase confidence threshold
4. **Close Other Apps**: Free up system resources

#### For Accessibility
1. **Increase Voice Volume**: Adjust in config
2. **Slow Down Speech**: Reduce rate in config
3. **Enable Fullscreen**: Set fullscreen: true in config
4. **Use High Contrast**: Dark theme is default

## 📊 Features in Detail

### AI Models
- **YOLO Object Detection**: Real-time detection of 80+ object classes
- **MiDaS Depth Estimation**: Accurate 3D depth mapping
- **Distance Calculation**: Physics-based distance estimation
- **Risk Assessment**: Multi-factor risk analysis

### Navigation System
- **Path Planning**: A* algorithm with obstacle avoidance
- **Safety Zones**: Configurable distance thresholds
- **Real-time Updates**: Continuous path recalculation
- **Emergency Routing**: Fastest path to exits

### Analytics & Logging
- **Detection Logging**: Comprehensive object detection history
- **Session Tracking**: Navigation session analytics
- **Performance Metrics**: FPS, detection counts, accuracy
- **Data Export**: JSON export of analytics data

### Accessibility Features
- **Voice Feedback**: Real-time audio navigation instructions
- **Gesture Control**: Hand gesture recognition
- **Emergency Mode**: Quick activation for urgent situations
- **Multi-modal Input**: Voice, gesture, and UI controls

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Install** dependencies: `python install.py`
5. **Test** your changes: `python main_new.py`
6. **Submit** a pull request

### Code Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive error handling
- Include proper logging
- Write clear documentation
- Test thoroughly before submitting

### Adding New Features
1. **Create Module**: Add new module in appropriate directory
2. **Update Imports**: Add to `__init__.py` files
3. **Update Main**: Integrate into `main_new.py`
4. **Test**: Test thoroughly
5. **Document**: Update documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO**: Real-time object detection
- **MiDaS**: Depth estimation models
- **MediaPipe**: Hand gesture recognition
- **CustomTkinter**: Modern UI framework
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📞 Support

### Getting Help
1. **Check Documentation**: Review this README and `README_MODULAR.md`
2. **Check Logs**: Review `indoor_nav.log` for error details
3. **Test Components**: Use individual module tests
4. **Check Configuration**: Verify `config.json` settings

### Reporting Issues
When reporting issues, please include:
- **System Info**: OS, Python version, hardware specs
- **Error Logs**: Relevant log entries from `indoor_nav.log`
- **Steps to Reproduce**: Clear reproduction steps
- **Expected vs Actual**: What you expected vs what happened

### Community
- **GitHub Issues**: [Report bugs and request features](https://github.com/PathanWasim/SafeStep/issues)
- **Discussions**: [Join community discussions](https://github.com/PathanWasim/SafeStep/discussions)
- **Wiki**: [Check the wiki for additional documentation](https://github.com/PathanWasim/SafeStep/wiki)

## 🚀 Roadmap

### Planned Features
- [ ] **Indoor Mapping**: SLAM-based indoor mapping
- [ ] **Bluetooth Integration**: Connect to external sensors
- [ ] **Mobile App**: Companion mobile application
- [ ] **Cloud Analytics**: Remote analytics and monitoring
- [ ] **Multi-language**: Support for multiple languages
- [ ] **Offline Mode**: Work without internet connection

### Performance Improvements
- [ ] **Model Optimization**: Quantized models for faster inference
- [ ] **Memory Optimization**: Reduced memory footprint
- [ ] **Battery Optimization**: Power-efficient processing
- [ ] **Parallel Processing**: Multi-threaded operations

---

**🎉 SafeStep - Making indoor navigation accessible for everyone!**

*Built with ❤️ for the visually impaired community*