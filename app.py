import gradio as gr
import cv2
import numpy as np
from model import ASLDetector

detector = ASLDetector()

def detect_asl(image):
    """Process image and detect ASL gesture."""
    if image is None or not isinstance(image, np.ndarray):
        return None, "Please provide an image (use Upload or capture from Webcam)"

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
with gr.Blocks(title="ASL Hand Detection System") as demo:
    gr.Markdown("""
    # ASL Hand Detection System
    American Sign Language hand gesture detection using MediaPipe.

    **Supported Gestures:**
    - A: Closed fist
    - V: Peace sign (index and middle fingers extended)
    - B: All fingers extended, thumb tucked
    - 1: Index finger only extended
    - W: Index, middle, and ring fingers extended
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                sources=["upload", "webcam"],
                type="numpy",
                label="Input Image",
                interactive=True
            )
            submit_btn = gr.Button("Detect ASL Gesture", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Detected Hand Landmarks")
            output_text = gr.Textbox(label="Detection Result", lines=3)

    submit_btn.click(
        fn=detect_asl,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()