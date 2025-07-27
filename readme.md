# SafeStep - Indoor Navigation System

This repository contains only the essential files for running and understanding the SafeStep Indoor Navigation System. Sensitive, large, and user-specific files are excluded for security and efficiency.

## Included Files and Folders

- `main_new.py`         : Main application entry point (modular version)
- `install.py`          : Installation and setup script
- `requirements.txt`    : Python dependencies
- `README.md`           : This documentation file
- `README_MODULAR.md`   : Detailed modular architecture and usage
- `config.example.json` : Example configuration (copy to `config.json` and edit as needed)
- `ai_models/`          : AI and ML components (object detection, depth estimation)
- `models/`             : Data models and core data structures
- `sensors/`            : Camera, gesture, and voice assistant modules
- `database/`           : Database management code (no user data included)
- `navigation/`         : Path planning and navigation logic
- `ui/`                 : User interface components

## Excluded (via .gitignore)

- Model weights (`*.pt`)
- Training runs (`runs/`)
- Database files (`*.db`)
- Log files (`*.log`)
- Python cache (`__pycache__/`, `*.pyc`)
- IDE/project files (`.vscode/`, `.idea/`)
- User-specific config (`config.json`)

---

**To use this project:**
1. Copy `config.example.json` to `config.json` and edit as needed.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main_new.py`