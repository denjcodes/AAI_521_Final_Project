import os
import cv2
import numpy as np
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download

# Global model cache for lazy loading
_model_cache = {}

# Model configurations for different EfficientNet versions
MODEL_CONFIGS = {
    "EfficientNetB4": {
        "repo_id": "d2j666/asl-efficientnets",  # UPDATE THIS!
        "filename": "efficientnetb4_asl.h5",
        "input_size": (224, 224),
        "classes": 29,
        "description": "EfficientNetB4 - Balanced performance and speed"
    },
    "EfficientNetB7": {
        "repo_id": "d2j666/asl-efficientnets",  # UPDATE THIS!
        "filename": "efficientnetb7_asl.h5",
        "input_size": (224, 224),
        "classes": 29,
        "description": "EfficientNetB7 - Higher accuracy, slower inference"
    },
    "EfficientNetB9": {
        "repo_id": "d2j666/asl-efficientnets",  # UPDATE THIS!
        "filename": "efficientnetb9_asl.h5",
        "input_size": (224, 224),
        "classes": 29,
        "description": "EfficientNetB9 - Highest accuracy, slowest inference"
    }
}


class ASLDetectorML:
    """
    ASL hand gesture detection using trained EfficientNet models.

    This detector uses deep learning models trained on the ASL Alphabet dataset
    to classify 29 different gestures (A-Z, del, nothing, space).
    """

    # ASL class labels (29 total)
    LABELS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space'
    ]

    def __init__(self, model_name: str = "EfficientNetB4"):
        """
        Initialize the ML-based ASL detector.

        Args:
            model_name: Name of the model to use ("EfficientNetB4", "EfficientNetB7", or "EfficientNetB9")
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.model = None
        self.input_size = self.config["input_size"]

        print(f"[INFO] Initializing {model_name} detector...")
        self._load_model()

    def _load_model(self):
        """Load model from HuggingFace Hub with caching."""
        global _model_cache

        # Check if model is already cached in memory
        if self.model_name in _model_cache:
            print(f"[INFO] Loading {self.model_name} from memory cache")
            self.model = _model_cache[self.model_name]
            return

        try:
            print(f"[INFO] Downloading {self.model_name} from HuggingFace Hub...")
            print(f"[INFO] This may take 5-10 seconds on first load...")

            # Download model from HuggingFace Hub
            model_path = hf_hub_download(
                repo_id=self.config["repo_id"],
                filename=self.config["filename"],
                cache_dir="./models",  # Local cache directory
                token=os.environ.get("HF_TOKEN")  # Optional: for private repos
            )

            print(f"[INFO] Model downloaded to: {model_path}")
            print(f"[INFO] Loading model into memory...")

            # Load the Keras model
            self.model = load_model(model_path)

            # Cache the model for future use
            _model_cache[self.model_name] = self.model

            print(f"[INFO] {self.model_name} loaded successfully!")

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print(f"[ERROR] Make sure models are uploaded to HuggingFace Hub")
            print(f"[ERROR] Expected repo: {self.config['repo_id']}")
            print(f"[ERROR] Expected file: {self.config['filename']}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for EfficientNet model.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            Preprocessed image ready for model inference
        """
        # Resize to model's expected input size
        img = cv2.resize(image, self.input_size)

        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Assume it's already RGB from Gradio
            pass

        # Apply EfficientNet-specific preprocessing
        img = preprocess_input(img.astype(np.float32))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict ASL gesture from image.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            Tuple of (predicted_letter, confidence_score)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Preprocess image
        preprocessed = self.preprocess_image(image)

        # Run inference
        predictions = self.model.predict(preprocessed, verbose=0)[0]

        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        predicted_letter = self.LABELS[predicted_idx]

        return predicted_letter, confidence

    def process_frame(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
        """
        Process a single frame for ASL classification.

        This method maintains compatibility with the existing ASLDetector interface.

        Args:
            image: RGB image array

        Returns:
            Tuple of (annotated_image, predicted_letter, confidence)
        """
        try:
            # Run prediction
            letter, confidence = self.predict(image)

            # Create annotated image with prediction
            annotated_image = image.copy()

            # Add text overlay
            if confidence > 0.3:  # Only show if reasonably confident
                text = f"{letter} ({confidence:.2f})"
                cv2.putText(
                    annotated_image,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            return annotated_image, letter, confidence

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return image, None, None

    def close(self):
        """Release resources. Models stay in cache for reuse."""
        print(f"[INFO] {self.model_name} detector closed (model remains in cache)")


def get_available_models():
    """Get list of available model names."""
    return list(MODEL_CONFIGS.keys())


def get_model_info(model_name: str) -> dict:
    """Get configuration info for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found")
    return MODEL_CONFIGS[model_name]


def clear_model_cache():
    """Clear the global model cache to free memory."""
    global _model_cache
    _model_cache.clear()
    print("[INFO] Model cache cleared")