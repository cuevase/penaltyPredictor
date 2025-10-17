# Penalty Predictor Setup Guide

## Prerequisites
- Python 3.11.x (required for MediaPipe compatibility)
- pyenv (recommended for Python version management)

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd penaltyPredictor
```

### 2. Set up Python 3.11 (if using pyenv)
```bash
# Install Python 3.11 if not already installed
pyenv install 3.11.10

# Set local Python version for this project
pyenv local 3.11.10
```

### 3. Create and activate virtual environment
```bash
# Create virtual environment
python -m venv venv_penalty

# Activate virtual environment
# On macOS/Linux:
source venv_penalty/bin/activate
# On Windows:
# venv_penalty\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify installation
```bash
python -c "import cv2, mediapipe as mp, numpy as np, yt_dlp; from moviepy import VideoFileClip; print('✅ All packages imported successfully!')"
```

## Usage

Always activate the virtual environment before running scripts:
```bash
source venv_penalty/bin/activate  # macOS/Linux
# venv_penalty\Scripts\activate   # Windows

# Run data collection
python src/data_collection.py

# Run pose extraction
python src/pose_extraction.py --video path/to/your/video.mp4
```

## Notes
- MediaPipe requires Python 3.11.x (does not support Python 3.13 yet)
- Virtual environment is not committed to git (see .gitignore)
- Use `requirements.txt` for dependency management
