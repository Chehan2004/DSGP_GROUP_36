import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class TeaDetector:
    """
    Object detector for tea leaves using YOLO v11 neural network.

    This class encapsulates YOLO model loading and inference, providing
    functionality to detect tea leaf regions in images and generate
    corresponding crop regions with padding.

    Attributes
    ----------
    model : YOLO
        Loaded YOLO model instance
    padding : int
        Number of pixels to add around bounding boxes during cropping
    """

    MODEL_PATH = "models/yolo_model/YOLO-v11-nano-LeafDetection-v1.pt"

    def __init__(self, padding: int = 15) -> None:
        """
        Initialize TeaDetector with YOLO model.

        Parameters
        ----------
        padding : int, optional
            Padding in pixels to add around detected bounding boxes (default: 15)

        Raises
        ------
        FileNotFoundError
            If model file does not exist at MODEL_PATH
        """
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at: {os.path.abspath(self.MODEL_PATH)}"
            )
        self.model = YOLO(self.MODEL_PATH)
        self.padding = padding

    def detect_and_crop(
        self,
        image: Image.Image,
        conf_threshold: float = 0.3
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect tea leaves in image and generate cropped regions.

        Parameters
        ----------
        image : Image.Image
            Input image as PIL Image object (RGB mode)
        conf_threshold : float, optional
            Confidence threshold for YOLO detections (default: 0.3)

        Returns
        -------
        annotated_img : np.ndarray
            RGB image array with bounding boxes drawn
        crops : List[Dict]
            List of crop dictionaries, each containing:
            - 'image': PIL.Image crop
            - 'confidence': float confidence score
            - 'bounding_box': tuple (x1, y1, x2, y2)
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        results = self.model(img_array, conf=conf_threshold, verbose=False)
        annotated_img = img_array.copy()
        crops = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                px1 = max(0, x1 - self.padding)
                py1 = max(0, y1 - self.padding)
                px2 = min(width, x2 + self.padding)
                py2 = min(height, y2 + self.padding)

                crop_arr = img_array[py1:py2, px1:px2]

                if crop_arr.size > 0:
                    crops.append({
                        'image': Image.fromarray(crop_arr),
                        'confidence': conf,
                        'bounding_box': (x1, y1, x2, y2)
                    })

        return annotated_img, crops
