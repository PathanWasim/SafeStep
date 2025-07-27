#!/usr/bin/env python3
"""
Installation Script for Advanced Indoor Navigation System
========================================================

This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
    return True

def install_pytorch():
    """Install PyTorch with appropriate CUDA support"""
    print("🔍 Detecting system for PyTorch installation...")
    
    system = platform.system()
    if system == "Windows":
        # For Windows, install CPU version by default
        # Users can manually install CUDA version if needed
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    else:
        # For Linux/Mac, try to detect CUDA
        try:
            result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
            if result.returncode == 0:
                print("🖥️  CUDA detected, installing PyTorch with CUDA support...")
                command = "pip install torch torchvision torchaudio"
            else:
                print("🖥️  No CUDA detected, installing CPU version...")
                command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        except:
            print("🖥️  CUDA detection failed, installing CPU version...")
            command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(command, "Installing PyTorch")

def install_requirements():
    """Install requirements from requirements.txt"""
    if Path("requirements.txt").exists():
        return run_command("pip install -r requirements.txt", "Installing requirements")
    else:
        print("⚠️  requirements.txt not found, installing core packages...")
        
        # Install core packages
        packages = [
            "ultralytics>=8.0.0",
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.0",
            "pillow>=9.0.0",
            "SpeechRecognition>=3.10.0",
            "pyttsx3>=2.90",
            "customtkinter>=5.2.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "websockets>=11.0",
            "requests>=2.31.0",
            "aiohttp>=3.8.0",
            "dataclasses-json>=0.6.0"
        ]
        
        for package in packages:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"⚠️  Failed to install {package}, continuing...")

def install_system_dependencies():
    """Install system-specific dependencies"""
    system = platform.system()
    
    if system == "Linux":
        print("🐧 Installing Linux dependencies...")
        commands = [
            "sudo apt update",
            "sudo apt install -y python3-dev python3-pip",
            "sudo apt install -y portaudio19-dev python3-pyaudio",
            "sudo apt install -y espeak espeak-data libespeak1 libespeak-dev",
            "sudo apt install -y ffmpeg"
        ]
        
        for command in commands:
            if not run_command(command, f"Running: {command}"):
                print(f"⚠️  Failed to run: {command}")
        
        # Install pyaudio for Python
        run_command("pip install pyaudio", "Installing pyaudio")
        
    elif system == "Darwin":  # macOS
        print("🍎 Installing macOS dependencies...")
        commands = [
            "brew install portaudio",
            "pip install pyaudio"
        ]
        
        for command in commands:
            if not run_command(command, f"Running: {command}"):
                print(f"⚠️  Failed to run: {command}")
    
    elif system == "Windows":
        print("🪟 Windows detected - no additional system dependencies needed")

def create_config_file():
    """Create default configuration file"""
    config_content = {
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
    
    import json
    with open("config.json", "w") as f:
        json.dump(config_content, f, indent=2)
    
    print("✅ Configuration file created: config.json")

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics YOLO"),
        ("customtkinter", "CustomTkinter"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("sqlite3", "SQLite3"),
        ("json", "JSON"),
        ("threading", "Threading"),
        ("datetime", "DateTime"),
        ("pathlib", "Pathlib"),
        ("typing", "Typing"),
        ("enum", "Enum"),
        ("dataclasses", "Dataclasses"),
        ("logging", "Logging"),
        ("warnings", "Warnings")
    ]
    
    failed_imports = []
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def main():
    """Main installation function"""
    print("🚀 Advanced Indoor Navigation System - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch
    if not install_pytorch():
        print("⚠️  PyTorch installation failed, but continuing...")
    
    # Install requirements
    install_requirements()
    
    # Install system dependencies
    install_system_dependencies()
    
    # Create config file
    if not Path("config.json").exists():
        create_config_file()
    else:
        print("✅ Configuration file already exists")
    
    # Test imports
    print("\n" + "=" * 60)
    print("🧪 Testing installation...")
    test_imports()
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed!")
    print("\n📋 Next steps:")
    print("1. Run 'python main_new.py' to start the application")
    print("2. Make sure you have a webcam connected")
    print("3. Check the config.json file for settings")
    print("4. For voice features, ensure microphone is working")
    print("\n📚 For more information, see readme.md")

if __name__ == "__main__":
    main() 