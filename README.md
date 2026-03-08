# Tea Leaf Detection System

A comprehensive machine learning system for detecting and classifying tea leaves from images. The system extracts color, texture, and shape features, trains multiple models with strong regularization to prevent overfitting, and provides real learning curves and performance analysis. Trained models can be saved as `.pt` files for reuse without retraining.

## Features

- **Feature Extraction**  
  - Color features (RGB, HSV, LAB channels, green dominance)  
  - Texture features (GLCM, LBP, statistical measures)  
  - Shape/size features (contour properties, Hu moments, aspect ratio, etc.)

- **Multiple Machine Learning Models**  
  - Support Vector Machine (SVM)  
  - Decision Tree (with pruning)  
  - K-Nearest Neighbors (with enhanced regularization)  
  - Neural Network (MLP with early stopping and strong L2)  
  - Random Forest (with limitations)

- **Anti‑Overfitting Measures**  
  - Dataset balancing (undersampling majority class)  
  - PCA dimensionality reduction (15–30 components)  
  - Controlled noise and label flipping augmentation  
  - Conservative hyperparameters (shallow trees, more neighbors)  
  - REAL learning curves using `sklearn.model_selection.learning_curve`

- **Performance Analysis**  
  - Balanced accuracy, confusion matrices, ROC‑AUC  
  - Overfitting severity classification (minimal / mild / moderate / severe)  
  - Model comparison with ranking  
  - Learning curve plots (accuracy and loss)  
  - Feature importance (Random Forest)

- **Model Persistence**  
  - Save trained models, scalers, and PCA as `.pt` (PyTorch) files  
  - Reuse models without retraining  
  - Simple prediction function for single images

- **Validation Module**  
  - Evaluate all trained models on a separate validation folder  
  - Generate confusion matrices, classification reports, and per‑image predictions  
  - McNemar statistical test for pairwise model comparison (if true labels exist)

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy, pandas, SciPy
- scikit-learn
- scikit-image
- matplotlib, seaborn
- PyTorch (torch)
- joblib

A typical `requirements.txt`:
```
opencv-python
numpy
pandas
scipy
scikit-learn
scikit-image
matplotlib
seaborn
torch
joblib
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Chehan2004/DSGP_GROUP_36.git
   cd tea_leaf_recognition
   ```

2. Install the required packages (see above).

3. Prepare your dataset (see next section).

## Dataset Preparation

The system expects two folders for training:  
- `data/tea_leaves/` – contains images of tea leaves (label = 1)  
- `data/non_tea/` – contains images of other objects / leaves (label = 0)

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

If these folders do not exist, the script will create them and ask you to add images.

For validation, you can provide a folder (e.g., `data/validation/`) with any of the following structures:
- Subfolders named `tea_leaves` / `tea_leaves_test` (label 1) and `non_tea` / `non_tea_test` (label 0) – metrics will be computed.
- A single folder with images (labels unknown) – only per‑image predictions are saved.
- A single subfolder (e.g., `validation_images`) – images are treated as unlabeled.

## Usage

Run the main script:

```bash
python tea_leaf_detection.py
```

You will be prompted with three options:

### 1. Train models on tea leaf dataset
Trains all five models using the images in `data/tea_leaves` and `data/non_tea`.  
All results are saved in the `results/` folder:
- Feature CSV files (`tea_leaf_features.csv`, `feature_importance_scores.csv`, `top_20_features.csv`)
- Feature importance plot (`feature_importance.png`)
- For each model:
  - Performance analysis plots and CSV
  - REAL learning curves (accuracy and loss)
  - Model saved as `.pt` (e.g., `svm_trained_model.pt`)
- Model comparison table and visualisation
- Reuse instructions (`reuse_instructions.txt`)

### 2. Test a single image
Provide the path to an image. The script will:
- Extract features using the same pipeline
- Load all trained `.pt` models from `results/`
- Display predictions and confidence scores
- Save a CSV with predictions in `results/predictions/`

### 3. Validate on validation folder
Specify a folder containing validation images. The script will:
- Load all trained models from `results/`
- Extract features and make predictions
- If true labels are available (via subfolder names), compute metrics and confusion matrices
- Save per‑image predictions and a validation summary
- Perform McNemar pairwise tests and generate a heatmap

## Output and Results

After training, the `results/` folder contains:

| File / Folder | Description |
|---------------|-------------|
| `tea_leaf_features.csv` | Extracted features for all training images |
| `feature_importance_scores.csv` | Feature importance (Random Forest) |
| `top_20_features.csv` | Only the 20 most important features |
| `feature_importance.png` | Bar plot of top 15 features |
| `*_real_learning_curves.png` | REAL learning curves for each model |
| `*_performance_analysis.png` | Performance metrics and overfitting analysis |
| `*_learning_curve_data.csv` | Raw data for learning curves |
| `model_comparison_balanced.csv` | Comparison table with balanced metrics |
| `model_comparison_balanced_visualization.png` | Visual comparison |
| `*_trained_model.pt` | Trained model (with scaler and PCA) ready for reuse |
| `reuse_instructions.txt` | Quick guide to reuse models |
| `knn_final_params.csv` | Final KNN parameters |
| `neural_network_architecture.csv` | NN architecture details |
| `neural_network_training_history.png` | Training loss and validation score curves (for NN) |
| `predictions/` | CSV files for single‑image predictions |
| `validation/` | Validation results (if option 3 is used) |

### .pt Model Files

Each `.pt` file is a dictionary containing:
- `model`: the trained scikit‑learn estimator
- `scaler`: the `StandardScaler` used (if any)
- `pca`: the PCA transformer (if any)
- `model_type`: e.g., `svm`, `decision_tree`
- Additional metadata (`classes`, `n_features_in`, `timestamp`, etc.)

## Reusing Trained Models Without Retraining

You can load any saved model and make predictions on new images without re‑running the training pipeline.

**Quick example:**

```python
import torch
import cv2
import numpy as np
from tea_leaf_detection import TeaLeafFeatureExtractor

# Load model data
model_data = torch.load('results/svm_trained_model.pt', weights_only=False)
model = model_data['model']
scaler = model_data['scaler']
pca = model_data['pca']

# Extract features from a new image (same as during training)
extractor = TeaLeafFeatureExtractor()
img = cv2.imread('new_leaf.jpg')
# ... (preprocess as in predict_single_image)
features = ...  # shape (1, n_features)

# Apply preprocessing
if scaler:
    features = scaler.transform(features)
if pca:
    features = pca.transform(features)

# Predict
pred = model.predict(features)
prob = model.predict_proba(features)   # if available
```

See `reuse_instructions.txt` for more details.

## Folder Structure

```
tea-leaf-detection/
├── tea_leaf_detection.py          # Main script
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── data/                           # Training data (created automatically)
│   ├── tea_leaves/                 # Place tea leaf images here
│   └── non_tea/                    # Place non-tea images here
├── validation/                      # Optional validation folder
│   ├── tea_leaves/                  # (if labels known)
│   └── non_tea/                     # (if labels known)
└── results/                         # All outputs (created after training)
    ├── svm_trained_model.pt
    ├── decision_tree_trained_model.pt
    ├── knn_trained_model.pt
    ├── neural_network_trained_model.pt
    ├── random_forest_trained_model.pt
    ├── ... (other files)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Feature extraction methods inspired by common image processing techniques for plant recognition.
- Regularisation strategies adapted from scikit‑learn documentation and best practices for avoiding overfitting on small datasets.
