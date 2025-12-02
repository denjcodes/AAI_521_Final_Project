import gradio as gr
import cv2
import numpy as np
from model import ASLDetector
from model_ml import ASLDetectorML

# Global detector cache for lazy loading
_detector_cache = {}


def get_detector(model_choice):
    """Get or create detector instance with lazy loading and caching."""
    global _detector_cache

    # Check if detector is already cached
    if model_choice in _detector_cache:
        return _detector_cache[model_choice]

    # Create new detector instance
    print(f"[INFO] Creating new detector: {model_choice}")

    detector = ASLDetector() if model_choice == "MediaPipe (Rule-based)" else ASLDetectorML(model_name=model_choice)

    # Cache for future use
    _detector_cache[model_choice] = detector

    return detector


def detect_asl(image, model_choice):
    """Process image and detect ASL gesture using selected model."""
    print(f"[INFO] detect_asl called - model: {model_choice}, image type: {type(image)}, is None: {image is None}")

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

    try:
        # Get or create detector (lazy loading)
        detector = get_detector(model_choice)

        # Process image
        annotated_image, letter, confidence = detector.process_frame(image)
        print(f"[INFO] Detection result - letter: {letter}, confidence: {confidence}")

        # Create result message
        if letter and letter != "Unknown":
            result = f"Detected: {letter} (Confidence: {confidence:.2f})\nModel: {model_choice}"
        elif letter == "Unknown":
            if model_choice == "MediaPipe (Rule-based)":
                result = "Hand detected but gesture not recognized. Try: A, V, B, 1, or W"
            else:
                result = f"Hand detected but gesture not recognized.\nModel: {model_choice}"
        else:
            result = "No hand detected. Please show a clear hand gesture."

        print(f"[INFO] Returning result: {result}")
        return annotated_image, result

    except Exception as e:
        error_msg = f"Error loading model: {str(e)}\n\nPlease ensure models are uploaded to HuggingFace Hub.\nSee MODEL_SETUP.md for instructions."
        print(f"[ERROR] {error_msg}")
        return image, error_msg


# Create Gradio interface with tabs for different input methods
with gr.Blocks(title="ASL Hand Detection System") as demo:
    gr.Markdown("""
    # ASL Hand Detection System
    American Sign Language hand gesture detection using MediaPipe and Deep Learning.

    - **EfficientNetB4**: Balanced performance and speed (recommended)
    - **EfficientNetB7**: Higher accuracy, slower inference
    - **EfficientNetB9**: Highest accuracy, slowest inference
    - **MediaPipe (Rule-based)**: Fast, lightweight fallback (5 gestures only)

    **Supported Gestures (ML Models):** A-Z, del, nothing, space (29 total)

    **MediaPipe Gestures:** A, V, B, 1, W (5 total)
    """)

    # Model selector dropdown
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=[
                "EfficientNetB4",
                "EfficientNetB7",
                "EfficientNetB9",
                "MediaPipe (Rule-based)"
            ],
            value="MediaPipe (Rule-based)",
            label="Select Model",
            info="First-time model (EfficientNet Based) loading may take 5-10 seconds"
        )

    gr.Markdown("**Note:** Switching between ML models (B4/B7/B9) may take 5-10 seconds on first load as the model downloads from HuggingFace Hub. Subsequent uses will be instant.")

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
                inputs=[webcam_input, model_selector],
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
                inputs=[upload_input, model_selector],
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
                inputs=[stream_input, model_selector],
                outputs=[stream_output, stream_result]
            )

if __name__ == "__main__":
    try:
        print("[INFO] Starting ASL Hand Detection System...")
        print("[INFO] Note: First-time model loading may take 5-10 seconds")
        demo.launch()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down gracefully...")
    finally:
        print("[INFO] Application stopped")