**🧭 SafeStep - Advanced Indoor Navigation System**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI-Powered Indoor Navigation System for Visually Impaired and Accessibility**

SafeStep provides real-time, AI-driven indoor navigation assistance tailored for visually impaired users and those requiring additional accessibility support. Leveraging computer vision, deep learning, and multi-modal interfaces, SafeStep guides users safely through complex indoor spaces.

---

## 📋 Table of Contents

1. [Key Features](#key-features)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Detailed Features](#detailed-features)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)
13. [Support](#support)
14. [Roadmap](#roadmap)

---

## 🔑 Key Features

### 🤖 AI-Powered Detection

* **Real-time Object Detection**: YOLO-based detection of people, obstacles, doors, stairs, and furniture.
* **Depth Estimation**: MiDaS model for accurate 3D depth mapping.
* **Risk Assessment**: Multi-factor classification into High, Medium, Low, or Minimal risk zones.
* **Distance Calculation**: Physics-based estimation of object distances.

### 🎤 Multi-Modal Accessibility

* **Voice Assistant**: Natural language processing for voice commands and feedback.
* **Gesture Recognition**: MediaPipe-powered hand gesture controls.
* **Text-to-Speech**: Real-time audio instructions and environment descriptions.
* **Emergency Mode**: One-command activation for critical situations.

### 🧭 Advanced Navigation

* **A* Path Planning*\*: Efficient route computation with obstacle avoidance.
* **Real-time Analytics**: Continuous logging of detections and navigation metrics.
* **Safety Monitoring**: Dynamic alerts based on configurable distance thresholds.
* **Multiple Modes**: Autonomous, Guided, Exploration, and Emergency modes.

### 🖥️ Modern User Interface

* **Dark Theme**: Sleek CustomTkinter interface by default.
* **Live Visualization**: Video feed overlays highlighting detections.
* **Analytics Dashboard**: Exportable statistics and session logs.
* **Responsive Layout**: Scales across different screen sizes.

---

## 💻 System Requirements

| Component             | Minimum             | Recommended         |
| --------------------- | ------------------- | ------------------- |
| Python                | 3.8+                | 3.9+                |
| RAM                   | 4 GB                | 8 GB+               |
| Camera                | Any USB/webcam      | 720p or higher      |
| Microphone & Speakers | Required            | High-quality device |
| GPU (optional)        | CPU inference       | CUDA-enabled GPU    |
| OS                    | Windows/Linux/macOS | Windows/Linux/macOS |

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/PathanWasim/SafeStep.git
   cd SafeStep
   ```

2. **Install dependencies**

   ```bash
   python install.py
   ```

   > This script will:
   >
   > * Verify Python version
   > * Install PyTorch (with CUDA if available)
   > * Install other Python packages
   > * Generate default `config.json`
   > * Validate imports

3. **Run the application**

   * **Modular version (recommended)**:

     ```bash
     python main_new.py
     ```
   * **Original version**:

     ```bash
     python main.py
     ```

---

## 📁 Project Structure

```plaintext
SafeStep/
├── main_new.py            # 🚀 Modular entry point
├── main.py                # Monolithic version
├── install.py             # 🔧 Automated installer
├── requirements.txt       # 📦 Python dependencies
├── config.json            # ⚙️ Default configuration
├── README.md              # 📖 Project documentation
├── README_MODULAR.md      # 🗂️ Modular architecture details
├── LICENSE                # 📜 MIT License
├── ai_models/             # 🤖 AI components
│   ├── depth_estimator.py # MiDaS depth mapping
│   └── object_detector.py # YOLO-based detection
├── sensors/               # 📡 Hardware interfaces
│   ├── camera_manager.py  # Camera input handling
│   ├── gesture_controller.py # MediaPipe gestures
│   └── voice_assistant.py # Speech recognition & TTS
├── navigation/            # 🧭 Path planning & safety
│   └── path_planner.py    # A* implementation
├── database/              # 💾 Persistence layer
│   └── database_manager.py# SQLite operations
├── models/                # 📊 Data structures & enums
│   └── data_models.py     # Core classes
├── ui/                    # 🖥️ User interface
│   └── main_window.py     # CustomTkinter components
└── Reports/               # 📊 Research and documentation
    ├── Object Detection final.pptx
    └── Research Paper .pdf
```

---

## ⚙️ Configuration

Edit `config.json` to customize camera settings, detection thresholds, navigation distances, and UI options:

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

---

## 🎮 Usage Guide

### Voice Commands

* **Navigate to \[destination]** — e.g., "Navigate to bathroom"
* **Find exit** — Locate the nearest exit
* **Where am I?** — Describe current location
* **What do you see?** — Enumerate visible objects
* **Emergency** — Trigger emergency mode

### Gesture Controls

* **Open palm** — Stop or pause navigation
* **Pointing finger** — Move forward
* **Left-hand wave** — Turn left
* **Right-hand wave** — Turn right
* **Raised hand** — Request help

### UI Controls

* **Mode Selector** — Switch navigation modes
* **Confidence Slider** — Adjust detection sensitivity
* **Voice Toggle** — Enable/disable speech assistant
* **Emergency Button** — Immediate emergency activation
* **Export Data** — Save analytics as JSON

---

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**

   ```bash
   python install.py
   # or
   pip install -r requirements.txt
   ```

2. **Camera not detected**

   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera open:', cap.isOpened()); cap.release()"
   ```

3. **Voice/TTS failures**

   ```bash
   python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
   python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait()"
   ```

4. **CUDA/GPU issues**

   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

---

## ⛑️ Performance Optimization

* **Enable GPU**: Use CUDA-enabled PyTorch for inference.
* **Lower Resolution**: Decrease camera resolution in `config.json`.
* **Adjust Thresholds**: Tweak confidence and NMS values.
* **Close Background Apps**: Free up CPU/RAM resources.

---

## 📊 Detailed Features

### AI Models

* **YOLO**: Detects 80+ classes in real time.
* **MiDaS**: Generates depth maps for distance estimation.
* **Risk Analysis**: Combines depth, speed, and context.

### Navigation System

* **A\***: Finds optimal routes with dynamic re-planning.
* **Safety Zones**: Configurable per-user distance settings.
* **Emergency Routing**: Fastest path to nearest exit.

### Analytics & Logging

* **Detection Logs**: Timestamped object events.
* **Session Metrics**: Duration, distance traveled, errors.
* **Export**: Download logs as JSON for offline analysis.

### Accessibility

* **Speech Feedback**: Clear, natural TTS guidance.
* **Gesture Input**: Hands-free control via MediaPipe.
* **Emergency Mode**: Vocal and visual alerts.

---

## 🤝 Contributing

1. **Fork** the repo
2. **Clone** your fork
3. **Create** a feature branch
4. **Install** dependencies: `python install.py`
5. **Implement** and **Test** your changes
6. **Document** updates and **Submit** a Pull Request

**Code Guidelines:**

* Follow PEP8 standards
* Include error handling and logging
* Write unit tests
* Update documentation

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* **YOLO**: Real-time detection
* **MiDaS**: Depth estimation
* **MediaPipe**: Gesture recognition
* **CustomTkinter**: UI framework
* **OpenCV** & **PyTorch**: Core vision and ML libraries

---

## 📞 Support

* **Documentation**: Review this README and `README_MODULAR.md`
* **Logs**: Check `indoor_nav.log`
* **Issues**: Report at [GitHub Issues](https://github.com/PathanWasim/SafeStep/issues)
* **Discussions**: Join at [GitHub Discussions](https://github.com/PathanWasim/SafeStep/discussions)

---

## 🚀 Roadmap

* [ ] SLAM-based indoor mapping
* [ ] Bluetooth sensor integration
* [ ] Companion mobile app
* [ ] Cloud analytics dashboard
* [ ] Multi-language support
* [ ] Offline navigation mode

**SafeStep – Empowering accessible indoor navigation**
