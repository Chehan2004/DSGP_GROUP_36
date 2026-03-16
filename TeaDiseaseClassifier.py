import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


class TeaDiseaseClassifier:

    DISEASE_CLASSES = [
        'Anthracnose', 'Gray Blight', 'Green Mirid Bug', 'Heliopeltis',
        'Red Rust', 'Red Spider', 'Sunlight Scorching', 'Tea Leaf Blight',
        'Tea Red Leaf Spot', 'Tea Red Scab', 'Thrips', 'Algal Leaf Spot',
        'Bird Eye Spot', 'Brown Blight', 'Gray Light', 'White Spot', 'Healthy'
    ]

    MODEL_CONFIGS = {
        'EfficientNet': {
            'path': "models/disease_detection_v1/efficientnet_best.h5",
            'input_size': (224, 224)
        },
        'MobileNet': {
            'path': "models/disease_detection_v1/mobilenet.h5",
            'input_size': (160, 160)
        },
        'ResNet50': {
            'path': "models/disease_detection_v1/resnet50_best.h5",
            'input_size': (160, 160)
        }
    }

    def __init__(self) -> None:
        self.loaded_models: Dict[str, Dict[str, Any]] = {}

        for name, config in self.MODEL_CONFIGS.items():
            path = config['path']
            if os.path.exists(path):
                try:
                    self.loaded_models[name] = {
                        'model': load_model(path),
                        'input_size': config['input_size']
                    }
                except Exception as e:
                    print(f"Error loading model '{name}' from '{path}': {e}")


    def predict(self, image_input: Any) -> Tuple[str, float]:
            """
            Predict the disease class of a tea leaf using an ensemble of Keras models.

            Parameters
            ----------
            image_input : Any
                Cropped tea leaf image represented as a NumPy array with shape (H, W, C) in RGB format
                passed directly from the single-class leaf localization output. Can also be a
                PIL.Image.Image object or a file path string.

            Returns
            -------
            disease_class : str
                The predicted disease name determined by the ensemble average.
            confidence : float
                The confidence score of the prediction as a percentage (range 0.0 to 100.0).
            """
            if not self.loaded_models:
                raise RuntimeError("No models loaded. Please check model paths and loading process.")

            # Handle input types
            if isinstance(image_input, np.ndarray):
                # Assume RGB array from localization crop
                base_img = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                base_img = image_input
            elif isinstance(image_input, str):
                base_img = Image.open(image_input).convert('RGB')
            else:
                raise TypeError("Input must be a file path, PIL Image, or np.ndarray")

            all_predictions = []
            metadata = {}

            for model_name, model_info in self.loaded_models.items():
                try:
                    model = model_info['model']
                    input_size = model_info['input_size']

                    # Use the processed base_img for resizing
                    img_resized = base_img.resize(input_size)
                    img_array = keras_image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis = 0)
                    img_array = img_array / 255.0

                    raw_predictions = model.predict(img_array, verbose = 0)[0]
                    probabilities = tf.nn.softmax(raw_predictions).numpy()

                    # Append to list for ensemble averaging
                    all_predictions.append(probabilities)

                except Exception as e:
                    metadata[model_name] = {
                        'error': str(e),
                        'raw_logits': None,
                        'probabilities': None,
                    }

            if not all_predictions:
                raise RuntimeError("All model predictions failed. Check metadata for details.")

            ensemble_predictions = np.mean(all_predictions, axis = 0)
            predicted_idx = int(np.argmax(ensemble_predictions))
            confidence = float(ensemble_predictions[predicted_idx]) * 100.0
            disease_class = (
                self.DISEASE_CLASSES[predicted_idx]
                if predicted_idx < len(self.DISEASE_CLASSES)
                else "Unknown"
            )
            return disease_class, confidence


if __name__ == "__main__":
    # Path to the leaf image you want to classify
    test_image_path = "WhatsApp Image 2026-03-14 at 13.32.49.jpeg"

    if not os.path.exists(test_image_path):
        print(f"Error: Image not found at {test_image_path}")
    else:
        try:
            print("Initializing Disease Classifier (Loading EfficientNet, MobileNet, ResNet)...")
            classifier = TeaDiseaseClassifier()

            print(f"Analyzing image: {test_image_path}...")
            # The predict method handles resizing and normalization internally
            disease, confidence = classifier.predict(test_image_path)

            print("-" * 40)
            print("TEA DISEASE CLASSIFICATION REPORT")
            print("-" * 40)
            print(f"Identified Condition : {disease}")
            print(f"Confidence Level      : {confidence:.2f}%")
            print("-" * 40)

            if confidence < 50.0:
                print("Warning: Low confidence score. Consider manual inspection.")

        except Exception as e:
            print(f"An error occurred during classification: {e}")
