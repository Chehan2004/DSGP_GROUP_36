import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

class YOLOProcessor:
    """
    Localization Engine for the Tea Leaf Pipeline.
    
    This class wraps the YOLOv8/v10/v11 inference logic to detect, bound, 
    and extract individual leaf candidates from high-resolution field imagery.
    """

    def __init__(self, model_path: str):
        """
        Loads the pre-trained YOLO weights into memory.
        
        Args:
            model_path (str): Path to the .pt weight file.
        Raises:
            FileNotFoundError: If the model path is invalid.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO weights not found at: {model_path}")
        self.model = YOLO(model_path)

    def extract_leaf_crops(self, image_path: str) -> tuple[list, list, any]:
        """
        Performs object detection and generates image sub-regions (crops).
        
        Args:
            image_path (str): Path to the target image file.
            
        Returns:
            tuple: (
                list of numpy.ndarray: The cropped images,
                list of tuples: (x1, y1, x2, y2) coordinates,
                numpy.ndarray: The original RGB image for visualization
            )
        """
        # Path sanitation and BGR to RGB conversion
        clean_path = image_path.strip().replace('"', "").replace("'", "")
        img_bgr = cv2.imread(clean_path)
        
        if img_bgr is None:
            raise ValueError(f"CRITICAL: Failed to load image from {clean_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        # Inference - verbose=False reduces terminal noise
        results = self.model(img_rgb, verbose=False)
        
        crops, boxes = [], []

        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                # Extract coordinates as integers
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Defensive Clamp: Ensures coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Region of Interest (ROI) Slicing
                crop = img_rgb[y1:y2, x1:x2]
                
                if crop.size > 0:
                    crops.append(crop)
                    boxes.append((x1, y1, x2, y2))

        return crops, boxes, img_rgb

    def visualize_detections(self, original_img, boxes: list):
        """
        Overlays bounding boxes and indexing on the source image.
        
        Args:
            original_img (numpy.ndarray): The source RGB image.
            boxes (list): List of (x1, y1, x2, y2) coordinate tuples.
        """
        annotated = original_img.copy()
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Styling: Neon Green Bounding Boxes
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Labeling: Leaf index for tracking through the pipeline
            cv2.putText(annotated, f"ID:{i+1}", (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(annotated)
        plt.title(f"Stage 1: Localization Results ({len(boxes)} candidates)")
        plt.axis("off")
        plt.show()