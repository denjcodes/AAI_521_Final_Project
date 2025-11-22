import gradio as gr
import cv2
import numpy as np
from model import ASLDetector

detector = ASLDetector()

def detect_asl(image):
    """Process image and detect ASL gesture."""
    if image is None:
        return None, "Please provide an image"

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Process image
    annotated_image, letter, confidence = detector.process_frame(image)

    # Create result message
    if letter and letter != "Unknown":
        result = f"Detected: {letter} (Confidence: {confidence:.2f})"
    elif letter == "Unknown":
        result = "Hand detected but gesture not recognized. Try: A, V, B, 1, or W"
    else:
        result = "No hand detected. Please show a clear hand gesture."

    return annotated_image, result

# Create Gradio interface
demo = gr.Interface(
    fn=detect_asl,
    inputs=gr.Image(sources=["upload"], type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(label="Detected Hand Landmarks"),
        gr.Textbox(label="Detection Result", lines=3)
    ],
    title="ASL Hand Detection System",
    description="""
    American Sign Language hand gesture detection using MediaPipe.

    **Supported Gestures:**
    - A: Closed fist
    - V: Peace sign (index and middle fingers extended)
    - B: All fingers extended, thumb tucked
    - 1: Index finger only extended
    - W: Index, middle, and ring fingers extended

    Upload an image to detect ASL gestures!
    """,
    live=False
)

if __name__ == "__main__":
    demo.launch()