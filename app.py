import gradio as gr
import cv2
import numpy as np
from model import ASLDetector

detector = ASLDetector()

def detect_asl(image):
    """Process image and detect ASL gesture."""
    print(f"[INFO] detect_asl called - image type: {type(image)}, is None: {image is None}")

    if image is None or not isinstance(image, np.ndarray):
        print(f"[WARN] Invalid input - rejecting image")
        return None, "Please provide an image (use Upload or capture from Webcam)"

    print(f"[INFO] Image received - shape: {image.shape}, dtype: {image.dtype}")

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print(f"[INFO] Converted grayscale to RGB")
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        print(f"[INFO] Converted RGBA to RGB")

    # Process image
    annotated_image, letter, confidence = detector.process_frame(image)
    print(f"[INFO] Detection result - letter: {letter}, confidence: {confidence}")

    # Create result message
    if letter and letter != "Unknown":
        result = f"Detected: {letter} (Confidence: {confidence:.2f})"
    elif letter == "Unknown":
        result = "Hand detected but gesture not recognized. Try: A, V, B, 1, or W"
    else:
        result = "No hand detected. Please show a clear hand gesture."

    print(f"[INFO] Returning result: {result}")
    return annotated_image, result


# Create Gradio interface with tabs
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

    with gr.Tabs():
        with gr.Tab("Take a Picture"):
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam",
                        interactive=True
                    )
                    webcam_btn = gr.Button("Detect Gesture", variant="primary")

                with gr.Column():
                    webcam_output = gr.Image(label="Detected Hand Landmarks")
                    webcam_result = gr.Textbox(label="Detection Result", lines=3)

            webcam_btn.click(
                fn=detect_asl,
                inputs=webcam_input,
                outputs=[webcam_output, webcam_result]
            )

        with gr.Tab("Upload Image"):
            with gr.Row():
                with gr.Column():
                    upload_input = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        label="Upload Image",
                        interactive=True
                    )
                    upload_btn = gr.Button("Detect Gesture", variant="primary")

                with gr.Column():
                    upload_output = gr.Image(label="Detected Hand Landmarks")
                    upload_result = gr.Textbox(label="Detection Result", lines=3)

            upload_btn.click(
                fn=detect_asl,
                inputs=upload_input,
                outputs=[upload_output, upload_result]
            )

        with gr.Tab("Live Streaming"):
            with gr.Row():
                with gr.Column():
                    stream_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Live Webcam Feed",
                        interactive=True,
                        streaming=True
                    )

                with gr.Column():
                    stream_output = gr.Image(label="Detected Hand Landmarks")
                    stream_result = gr.Textbox(label="Detection Result", lines=3)

            stream_input.stream(
                fn=detect_asl,
                inputs=stream_input,
                outputs=[stream_output, stream_result]
            )

if __name__ == "__main__":
    try:
        print("[INFO] Starting ASL Hand Detection System...")
        demo.launch()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down gracefully...")
    finally:
        print("[INFO] Application stopped")