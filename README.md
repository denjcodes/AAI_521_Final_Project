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

## Currently Supported Gestures

- A: Closed fist
- V: Peace sign (index and middle fingers extended)
- B: All fingers extended, thumb tucked
- 1: Index finger only extended
- W: Index, middle, and ring fingers extended

## Setup

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Run the application
uv run python app.py

The application will be available at `http://localhots:7860`
```
## Testing

**Stop server:** `Ctrl+C`
```bash
# To kill previous instances of the run:
pkill -f "app.py" && sleep 1 && ps aux | grep "app.py" | grep -v grep

# Run the application
uv run python app.py
```

## Technical Stack

- **Hand Detection**: MediaPipe Hands (v0.10.9)
- **Classification**: Rule-based finger extension analysis
- **Frontend**: Gradio
- **Image Processing**: OpenCV, NumPy, CNN (TBD)
- **Package Management**: UV for reproducible builds
---