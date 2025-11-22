---
title: AAI 521 - ASL Hand Detection
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# AAI 521 - ASL Hand Detection System

American Sign Language hand gesture detection using MediaPipe, Gradio and CNN (TBD).

## Project Overview

This application detects and classifies basic static ASL gestures using MediaPipe Hands for hand landmark detection and a rule-based classifier for gesture recognition.

Features:
- Three input modes: Webcam snapshot, Image upload, and Live streaming
- Real-time hand landmark visualization
- Support for 5 ASL gestures (A, V, B, 1, W)
- Production logging for debugging

## Setup
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Run the application
uv run python app.py

The application will be available at `http://localhots:7860` (and can be killed gracefully with CTRL+C)
```

## Technical Stack

- **Hand Detection**: MediaPipe Hands (v0.10.9)
- **Classification**: Rule-based finger extension analysis
- **Frontend**: Gradio (with tabbed interface)
- **Image Processing**: OpenCV, NumPy, CNN (TBD)
- **Package Management**: UV for reproducible builds
- **Logging**: Production-ready [INFO]/[WARN] logging- 
---