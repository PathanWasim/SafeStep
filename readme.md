# SafeStep - Indoor Navigation System

🧭 **SafeStep** is an advanced indoor navigation system designed to help visually impaired individuals and provide enhanced navigation assistance in indoor environments.

## ✨ Features

### 🎥 **60 FPS Camera Feed**
- High-performance camera streaming at 60 FPS
- Real-time object detection and analysis
- Smooth video display with optimized processing

### 🔍 **Enhanced Object Detection**
The system can detect and provide navigation guidance for:

#### 🚪 **Exits & Entrances**
- Exit signs (red rectangular signs)
- Emergency exits
- Doors and entrances
- Stairs and escalators

#### 🚽 **Bathroom Facilities**
- Toilet detection
- Bathroom symbols
- Sink and mirror detection
- Restroom entrances

#### 🛗 **Elevator & Transportation**
- Elevator buttons (circular detection)
- Elevator doors
- Stair detection with safety warnings

#### 👥 **People & Safety**
- Person detection with distance warnings
- Safety equipment (fire extinguishers)
- Crowd detection

#### 🪑 **Furniture & Obstacles**
- Chairs, tables, and seating
- Electronic devices
- General furniture and obstacles

### 🎤 **Voice Assistant**
- Voice commands for navigation
- Text-to-speech feedback
- Emergency voice alerts

### 🖐️ **Gesture Control**
- Hand gesture recognition (when MediaPipe is available)
- Stop, forward, left, right, and help gestures

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam or USB camera
- Windows 10/11 (tested)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "SafeStep indoor navigation"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run SafeStep**
   ```bash
   python main_new.py
   ```

## 🎮 How to Use

### **Starting the System**
1. Run `python main_new.py`
2. The system will initialize all components
3. The main window will open with the camera feed

### **Camera Feed (60 FPS)**
- **Center Panel**: Live camera feed at 60 FPS
- **Real-time Detection**: Objects are highlighted with colored boxes
  - 🔴 **Red**: High-risk objects (stairs, people close by)
  - 🟠 **Orange**: Medium-risk objects (furniture, electronics)
  - 🟢 **Green**: Low-risk objects (toilets, exits, seating)

### **Navigation Modes**
- **Autonomous**: Full AI-powered navigation
- **Guided**: Voice-guided navigation
- **Exploration**: Free exploration mode
- **Emergency**: Emergency navigation to nearest exit

### **Voice Commands**
- **"Find bathroom"** or **"Find toilet"** - Locate nearest restroom
- **"Find exit"** - Locate nearest exit
- **"Find elevator"** - Locate elevator
- **"Where am I?"** - Get current location
- **"Help"** - Get assistance
- **"Stop"** - Stop navigation
- **"Emergency"** - Activate emergency mode

### **Control Panel**
- **Confidence Slider**: Adjust detection sensitivity (0.1 - 1.0)
- **Voice Toggle**: Enable/disable voice assistant
- **Mode Selector**: Change navigation mode
- **Emergency Button**: Quick emergency activation

### **Analytics Dashboard**
- **Real-time Stats**: Current detection statistics
- **Object Breakdown**: Count of detected objects by type
- **Safety Status**: Overall safety assessment
- **Export Data**: Save session data for analysis

## 🔧 Configuration

### Camera Settings (`config.json`)
```json
{
  "camera": {
    "device_id": 0,
    "resolution": [1280, 720],
    "fps": 60
  }
}
```

### Detection Settings
```json
{
  "detection": {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "model_path": "yolo11n.pt"
  }
}
```

## 🎯 Object Detection Capabilities

### **Automatic Detection**
The system automatically detects and provides guidance for:

| Object Type | Detection Method | Navigation Advice |
|-------------|------------------|-------------------|
| **Exit Signs** | Color detection (red) + Shape analysis | "🚪 Exit sign X.Xm ahead" |
| **Elevator Buttons** | Circle detection | "🛗 Elevator button X.Xm ahead" |
| **Bathroom Symbols** | Contour analysis | "🚽 Bathroom X.Xm ahead" |
| **Stairs** | Line detection (horizontal patterns) | "⚠️ Stairs X.Xm ahead - Use handrail" |
| **People** | YOLO detection | "👤 Person X.Xm ahead - Give space" |
| **Furniture** | YOLO detection | "🪑 [Object] X.Xm ahead - Navigate around" |

### **Distance Estimation**
- **Very Close** (< 0.5m): Immediate attention required
- **Close** (0.5-1.0m): Navigate carefully
- **Medium** (1.0-2.0m): Proceed with awareness
- **Far** (> 2.0m): Safe to proceed

## 🆘 Emergency Features

### **Emergency Mode**
- Press the **🚨 EMERGENCY** button
- System immediately searches for nearest exit
- Voice announces emergency procedures
- High-priority navigation to safety

### **Safety Warnings**
- **High-risk objects**: Immediate warnings for stairs, close people
- **Medium-risk objects**: Cautionary advice for furniture, electronics
- **Low-risk objects**: Informational guidance for facilities

## 📊 Performance

### **60 FPS Optimization**
- Optimized camera buffer settings
- Efficient frame processing
- Adaptive timing for consistent 60 FPS
- Real-time object detection without lag

### **System Requirements**
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: USB webcam or built-in camera
- **Storage**: 2GB free space for models

## 🛠️ Troubleshooting

### **Camera Not Working**
1. Check camera connection
2. Ensure no other applications are using the camera
3. Try different camera device ID in config.json
4. Restart the application

### **Low FPS**
1. Reduce resolution in config.json
2. Lower confidence threshold
3. Close other applications
4. Check system performance

### **Voice Assistant Issues**
1. Check microphone permissions
2. Ensure internet connection for speech recognition
3. Test microphone in system settings

### **Object Detection Issues**
1. Adjust confidence threshold
2. Ensure good lighting
3. Check camera focus
4. Update YOLO model if needed

## 🔮 Future Enhancements

- **Custom Model Training**: Train models for specific buildings
- **Indoor Mapping**: Create detailed indoor maps
- **Bluetooth Integration**: Connect to smart devices
- **Mobile App**: Companion mobile application
- **Multi-language Support**: International language support

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**SafeStep** - Making indoor navigation safer and more accessible for everyone! 🧭✨