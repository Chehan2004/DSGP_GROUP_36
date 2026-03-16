import os
from email.mime import image
from pyexpat import features
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import stats
from skimage import feature, measure


class TeaLeafPredictor:

    MODEL_PATHS = [
        'models/tea_leaf_rec/decision_tree_trained_model.pt',
        'models/tea_leaf_rec/knn_trained_model.pt',
        'models/tea_leaf_rec/neural_network_trained_model.pt',
        'models/tea_leaf_rec/random_forest_trained_model.pt',
        'models/tea_leaf_rec/svm_trained_model.pt'
    ]

    # Initialize the predictor by loading all models and setting the image size
    def __init__(self, image_size: Tuple[int, int] = (256, 256)) -> None:

        # Set the image size for preprocessing
        self.image_size = image_size
        self.models: List[Dict[str, Any]] = []
        self.model_names: List[str] = []

        for model_path in self.MODEL_PATHS:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            try:
                # Load the model and any associated preprocessing components (scaler, PCA)
                model_data = torch.load(model_path, weights_only=False)
                self.models.append({
                    'model': model_data['model'],
                    'scaler': model_data.get('scaler', None),
                    'pca': model_data.get('pca', None),
                    'path': model_path
                })
                # Extract the model name from the file path for later use
                model_name = os.path.basename(model_path).replace('_trained_model.pt', '').upper()
                self.model_names.append(model_name)

            # Handle any exceptions that occur during model loading and provide a clear error message
            except Exception as e:
                raise RuntimeError(f"No models loaded. Please check model paths and loading process. Error: {e}")

        if not self.models:
            raise ValueError("No models were loaded successfully")

    def _preprocess_image(self, image_input: Any) -> Dict[str, Any]:
            # Check for the numpy array from the YOLO crop (expected in RGB format)
            if isinstance(image_input, np.ndarray):
                # Convert RGB to BGR for OpenCV processing
                img = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, Image.Image):
                # Fixed typo: cvColor -> cvtColor
                img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(image_input)
                if img is None:
                    raise ValueError(f"Could not read image from path: {image_input}")

            img = cv2.resize(img, self.image_size)

            return {
                'rgb': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                'hsv': cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                'lab': cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
                'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            }

    def _extract_color_features(self, image: Dict[str, Any]) -> List[float]:
        features = []
        try:
            r, g, b = image['rgb'][:,:,0], image['rgb'][:,:,1], image['rgb'][:,:,2]
            green_ratio = np.mean(g)/(np.mean(r)+np.mean(b)+1e-8)
            green_dominance = np.mean(g)-np.mean(r)
            for channel in [r, g, b]:
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.median(channel),
                    stats.skew(channel.flatten()),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75)
                ])
            features.extend([green_ratio, green_dominance])

            h, s, v = image['hsv'][:,:,0], image['hsv'][:,:,1], image['hsv'][:,:,2]
            green_hue_mask = cv2.inRange(h, 30, 90)
            green_hue_ratio = np.sum(green_hue_mask>0)/(h.size+1e-8)
            features.extend([
                np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v), green_hue_ratio
            ])
            features.extend([np.mean(image['lab'][:,:,1]), np.mean(image['lab'][:,:,2])])

        except Exception:
            features = [0]*29

        return features

    def _extract_texture_features(self, image: Dict[str, Any]) -> List[float]:
        features = []
        try:
            gray = image['gray']
            glcm = feature.graycomatrix(gray, [1, 3], [0, np.pi/4], symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                values = feature.graycoprops(glcm, prop)
                features.extend(values.flatten())

            lbp = feature.local_binary_pattern(gray, 16, 2, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0,17))
            lbp_hist = lbp_hist.astype(float)/(lbp_hist.sum()+1e-8)
            features.extend(lbp_hist)

            features.extend([
                np.mean(gray), np.std(gray),
                stats.skew(gray.flatten()),
                stats.kurtosis(gray.flatten())
            ])
        except Exception:
            features = [0]*46
        return features

    def _extract_shape_features(self, image: Dict[str, Any]) -> List[float]:
        features = []
        try:
            gray = image['gray']
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                x, y, w, h = cv2.boundingRect(main_contour)
                bbox_area = w*h
                aspect_ratio = float(w)/h if h>0 else 0
                extent = area/bbox_area if bbox_area>0 else 0
                circularity = (4*np.pi*area)/(perimeter**2) if perimeter>0 else 0
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area/hull_area if hull_area>0 else 0
                equiv_diameter = np.sqrt(4*area/np.pi)

                if len(main_contour) >= 5:
                    ellipse = cv2.fitEllipse(main_contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis>0 else 0
                else:
                    major_axis, minor_axis, eccentricity = w, h, 0

                hu_moments = cv2.HuMoments(cv2.moments(main_contour)).flatten()

                features = [
                    area, perimeter, w, h, bbox_area, aspect_ratio, extent, circularity,
                    solidity, equiv_diameter, major_axis, minor_axis, eccentricity,
                    area/bbox_area, (perimeter**2)/(area+1e-8)
                ]
                features.extend(hu_moments[:7])
            else:
                features = [0]*22
        except Exception:
            features = [0]*22
        return features

    def _extract_all_features(self, image: Dict[str, Any]) -> np.ndarray:
        color_features = self._extract_color_features(image)
        texture_features = self._extract_texture_features(image)
        shape_features = self._extract_shape_features(image)
        all_features = color_features + texture_features + shape_features
        return np.array(all_features).reshape(1, -1)


    # Main prediction method that takes an image and returns the predicted label, confidence, and metadata for each model
    def predict(self, image: Any, voting_threshold: int = 3) -> Tuple[str, float, Dict[str, Dict[str, Any]]]:
            """
            Predict whether an input image is a tea leaf using an ensemble of models.

            Parameters
            ----------
            image : Any
                Cropped tea leaf image represented as a NumPy array with shape (H, W, C) in RGB format.
                Can also be a PIL.Image.Image object or a file path string.
            voting_threshold : int, optional
                The minimum number of models that must vote "Tea Leaf" for the final label
                to be considered "Tea Leaf" (default is 3).

            Returns
            -------
            label : str
                The final ensemble prediction: 'Tea Leaf' or 'Non-Tea'.
            confidence : float
                The confidence of the ensemble prediction as the fraction of models
                that voted for "Tea Leaf" (range 0.0 to 1.0).
            metadata : Dict[str, Dict[str, Any]]
                A dictionary containing per-model predictions and probabilities.
                Format:
                {
                    'MODEL_NAME': {'label': 0 or 1, 'probability': float or None},
                    ...
                }
                - 'label' is the predicted class by that model.
                - 'probability' is the model's probability of class 1 if available;
                otherwise, None or the label itself.
            """
            preprocessed = self._preprocess_image(image)
            features = self._extract_all_features(preprocessed)
            all_labels = []
            metadata = {}

            for i, model_data in enumerate(self.models):
                model = model_data['model']
                scaler = model_data['scaler']
                pca = model_data['pca']
                model_name = self.model_names[i]

                try:
                    X = features.copy()
                    if scaler is not None:
                        X = scaler.transform(X)
                    if pca is not None:
                        X = pca.transform(X)

                    label = model.predict(X)[0]
                    all_labels.append(int(label))

                    if hasattr(model, 'predict_proba'):
                        probability = float(model.predict_proba(X)[0,1])
                    else:
                        probability = float(label)

                    metadata[model_name] = {'label': int(label), 'probability': probability}
                except Exception as e:
                    metadata[model_name] = {'label': None, 'probability': None}

            vote_count = sum(all_labels)
            is_tea = vote_count >= voting_threshold
            confidence = vote_count / len(self.models)
            label = 'Tea Leaf' if is_tea else 'Non-Tea'

            return label, confidence, metadata


if __name__ == "__main__":
    # Path to the image you want to test
    test_image_path = "WhatsApp Image 2026-03-14 at 13.32.49.jpeg"

    # Check if the image exists before proceeding
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
    else:
        try:
            # Initialize the predictor
            # This will load the models from the predefined MODEL_PATHS
            predictor = TeaLeafPredictor(image_size=(256, 256))

            # Perform prediction
            # The predict method returns: label, confidence, and detailed metadata
            label, confidence, metadata = predictor.predict(test_image_path)

            # Display results
            print("-" * 30)
            print(f"Prediction Results for: {os.path.basename(test_image_path)}")
            print("-" * 30)
            print(f"Final Label: {label}")
            print(f"Confidence:  {confidence:.2%}")
            print("\nIndividual Model Votes:")

            for model_name, data in metadata.items():
                res = "Tea Leaf" if data['label'] == 1 else "Non-Tea"
                prob = data['probability']
                print(f"- {model_name:15}: {res} (Prob: {prob:.4f})")
            print("-" * 30)

        except Exception as e:
            print(f"An error occurred during testing: {e}")
