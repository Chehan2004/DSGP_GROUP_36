# QualityGrader.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt

class QualityGrader:
    """
    QualityGrader encapsulates the ResNet-based leaf classification pipeline.
    Provides methods to load a trained classifier, preprocess crops, classify,
    and visualize classification results.
    """

    def __init__(self, model_path: str, device: torch.device, num_classes: int = 5, class_map: dict = None):
        """
        Initialize the QualityGrader.

        Parameters:
            model_path (str): Path to the trained ResNet .pth file.
            device (torch.device): Device for inference ('cpu' or 'cuda').
            num_classes (int): Number of output classes in the classifier.
            class_map (dict): Optional mapping from class index to label string.
        """
        self.device = device
        self.num_classes = num_classes
        self.class_map = class_map if class_map is not None else {i: f"Class {i}" for i in range(num_classes)}
        self.model = self.load_resnet_classifier(model_path)

    def load_resnet_classifier(self, model_path: str):
        """
        Load a ResNet classifier and trained weights.

        Parameters:
            model_path (str): Path to the .pth checkpoint.

        Returns:
            torch.nn.Module: Loaded ResNet model in evaluation mode.
        """
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def get_preprocessing_pipeline():
        """
        Return a preprocessing pipeline to resize, normalize, and convert crops to tensors.
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def classify_crops(self, crops: list):
        """
        Classify a list of cropped leaf images.

        Parameters:
            crops (list): List of numpy arrays representing image crops.

        Returns:
            list: List of dictionaries containing 'crop', 'label', and 'confidence'.
        """
        preprocess = self.get_preprocessing_pipeline()
        results = []

        for crop in crops:
            input_tensor = preprocess(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
                confidence, pred_idx = torch.max(probs, dim=0)

            results.append({
                "crop": crop,
                "label": self.class_map[pred_idx.item()],
                "confidence": confidence.item()
            })

        return results

    @staticmethod
    def visualize_classification(results: list):
        """
        Visualize classification results with matplotlib.

        Parameters:
            results (list): Output of classify_crops().
        """
        if not results:
            print("No crops to classify.")
            return

        plt.figure(figsize=(4 * len(results), 5))

        for i, res in enumerate(results):
            ax = plt.subplot(1, len(results), i + 1)
            ax.imshow(res["crop"])
            ax.axis("off")
            ax.set_title(f"{res['label']}\n{res['confidence']:.1%}", fontsize=9)

        plt.tight_layout()
        plt.show()