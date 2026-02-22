# TeaDetector.py

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

class TeaDetector:
    """
    TeaDetector encapsulates the YOLO-based leaf detection pipeline.
    Provides methods for loading a YOLO model, preprocessing images,
    performing detection, cropping regions, and visualizing results.
    """

    def __init__(self, yolo_model_path: str, conf_threshold: float = 0.25, padding: int = 10):
        """
        Initialize the TeaDetector.

        Parameters:
            yolo_model_path (str): Path to YOLO .pt model file.
            conf_threshold (float): Minimum confidence threshold for keeping detections.
            padding (int): Number of pixels to add around bounding boxes when cropping.
        """
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found at: {yolo_model_path}")

        self.model = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.padding = padding

    def preprocess_image(self, image_path: str):
        """
        Load an image and convert from BGR to RGB.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            numpy.ndarray: RGB image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to read image at {image_path}. Check the file.")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def detect_and_crop(self, image_rgb):
        """
        Run YOLO detection on the image and extract cropped regions.

        Parameters:
            image_rgb (numpy.ndarray): Preprocessed RGB image.

        Returns:
            tuple:
                annotated (numpy.ndarray): Image with bounding boxes drawn.
                crops (list): List of cropped image regions.
        """
        height, width, _ = image_rgb.shape
        results = self.model(image_rgb, conf=self.conf_threshold)

        annotated = image_rgb.copy()
        crops = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Apply padding and clamp to image boundaries
                x1 = max(0, x1 - self.padding)
                y1 = max(0, y1 - self.padding)
                x2 = min(width, x2 + self.padding)
                y2 = min(height, y2 + self.padding)

                crop = image_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crops.append(crop)

                # Draw bounding box on annotated image
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return annotated, crops

    def visualize_detections(self, annotated, crops):
        """
        Display annotated image with detected crops.

        Parameters:
            annotated (numpy.ndarray): Image with bounding boxes.
            crops (list): Cropped image regions.

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(annotated)
        plt.axis("off")
        plt.title(f"Detected {len(crops)} Region(s)")
        plt.show()