import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# Hard-coded class mapping
# -----------------------------
CLASS_NAMES: Dict[int, str] = {
    0: "T0: Non-Tea / Waste",
    1: "T1: Premium Shoot",
    2: "T2: Standard Leaf",
    3: "T3: Mature Leaf",
    4: "T4: Low Grade",
}

class QualityGrader:

    def __init__(self, class_map: dict[int, str] = CLASS_NAMES) -> None:
        self.device = torch.device("cude" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map

        self.model = models.efficientnet_b0(weights = None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, len(class_map))

        state_dict = torch.load(
            "models/tea_leaf_assesment_model/EfficientNetB0_TeaAge_v2_5Class_20260301.pth",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        """

        Predict the tea leaf quality class for a given cropped image.

        This function performs inference using the trained EfficientNet model.
        The input image is first converted to a PIL image, then preprocessed
        using the same transformation pipeline used during model training.
        The processed image is passed through the neural network to obtain
        class probabilities. The class with the highest probability is
        selected as the prediction.

        Parameters
        ----------
        img : np.ndarray
            Cropped tea leaf image represented as a NumPy array
            with shape (H, W, C) in RGB format.

        Returns
        -------
        Tuple[str, float]
            label : str
                Human-readable class label corresponding to the predicted class.
            confidence : float
                Probability score of the predicted class (range 0.0 – 1.0).

        """
        image = Image.fromarray(img)
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probs = torch.softmax(output, dim = 1)
            conf, pred = torch.max(probs, 1)

        label = self.class_map[pred.item()]
        confidence = conf.item()

        return label, confidence

if __name__ == "__main__":
    import os

    import cv2

    # 1. Path to the leaf image you want to grade
    test_image_path = "WhatsApp Image 2026-03-14 at 13.32.49.jpeg"

    if not os.path.exists(test_image_path):
        print(f"Error: Image not found at {test_image_path}")
    else:
        try:
            # 2. Initialize the Quality Grader
            # This loads the EfficientNetB0 model and the pre-trained weights
            print("Loading QualityGrader model...")
            grader = QualityGrader()

            # 3. Load and prepare the image
            # OpenCV loads as BGR, so we must convert to RGB
            img_bgr = cv2.imread(test_image_path)
            if img_bgr is None:
                print("Error: Could not decode the image.")
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # 4. Perform Grading
                print(f"Grading image: {test_image_path}...")
                label, confidence = grader.predict(img_rgb)

                # 5. Output Results
                print("-" * 40)
                print("TEA QUALITY GRADING RESULT")
                print("-" * 40)
                print(f"Assigned Grade : {label}")
                print(f"Confidence     : {confidence:.2%}")
                print("-" * 40)

        except Exception as e:
            print(f"An error occurred during grading: {e}")
