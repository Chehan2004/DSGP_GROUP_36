import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage import feature, filters, measure
import os
import matplotlib

# Set the backend BEFORE importing pyplot
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
import traceback
import glob  # for validation

# imports for SVM and utilities
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, \
    balanced_accuracy_score
import joblib
from sklearn.base import clone

# import Decision Tree, KNN, and Neural Network
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA

# Import torch for .pt file creation
import torch
import torch.nn as nn
import pickle

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class TeaLeafFeatureExtractor:
    """
    Feature extraction specifically designed for tea leaf recognition
    """

    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.feature_names = []

    def load_and_preprocess_images(self, image_folder, label):
        """
        Load images from folder and preprocess them
        label: 1 for tea leaves, 0 for non-tea leaves
        """
        images_data = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        # Check if folder exists
        if not os.path.exists(image_folder):
            print(f"Error: Folder '{image_folder}' does not exist!")
            return images_data

        print(f"Loading images from: {image_folder}")

        try:
            for filename in sorted(os.listdir(image_folder)):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(image_folder, filename)

                    try:
                        # Read and preprocess image
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Warning: Could not read image {filename}")
                            continue

                        # Resize for consistency
                        img = cv2.resize(img, self.image_size)

                        # Convert to different color spaces
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        images_data.append({
                            'rgb': img_rgb,
                            'hsv': img_hsv,
                            'lab': img_lab,
                            'gray': img_gray,
                            'label': label,
                            'filename': filename
                        })

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        continue

            print(f"Loaded {len(images_data)} images with label {label}")

        except Exception as e:
            print(f"Error accessing folder {image_folder}: {e}")

        return images_data

    def extract_tea_specific_color_features(self, image):
        """
        Extract color features specific to tea leaves
        """
        features = []
        feature_names = []

        try:
            # RGB features
            r_channel = image['rgb'][:, :, 0]
            g_channel = image['rgb'][:, :, 1]
            b_channel = image['rgb'][:, :, 2]

            # Green dominance (tea leaves are typically green)
            green_ratio = np.mean(g_channel) / (np.mean(r_channel) + np.mean(b_channel) + 1e-8)
            green_dominance = np.mean(g_channel) - np.mean(r_channel)

            # Color statistics for each channel
            for i, channel in enumerate([r_channel, g_channel, b_channel]):
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.median(channel),
                    stats.skew(channel.flatten()),
                    np.percentile(channel, 25),  # Q1
                    np.percentile(channel, 75),  # Q3
                ])
                feature_names.extend([
                    f'rgb_{["R", "G", "B"][i]}_mean',
                    f'rgb_{["R", "G", "B"][i]}_std',
                    f'rgb_{["R", "G", "B"][i]}_median',
                    f'rgb_{["R", "G", "B"][i]}_skewness',
                    f'rgb_{["R", "G", "B"][i]}_Q1',
                    f'rgb_{["R", "G", "B"][i]}_Q3',
                ])

            features.extend([green_ratio, green_dominance])
            feature_names.extend(['green_ratio', 'green_dominance'])

            # HSV features
            h_channel = image['hsv'][:, :, 0]
            s_channel = image['hsv'][:, :, 1]
            v_channel = image['hsv'][:, :, 2]

            # Tea leaves typically have hue in green range (30-90 in OpenCV)
            green_hue_mask = cv2.inRange(h_channel, 30, 90)
            green_hue_ratio = np.sum(green_hue_mask > 0) / (h_channel.size + 1e-8)

            hsv_features = [
                np.mean(h_channel), np.std(h_channel),
                np.mean(s_channel), np.std(s_channel),
                np.mean(v_channel), np.std(v_channel),
                green_hue_ratio
            ]
            hsv_names = [
                'hue_mean', 'hue_std',
                'saturation_mean', 'saturation_std',
                'value_mean', 'value_std',
                'green_hue_ratio'
            ]

            features.extend(hsv_features)
            feature_names.extend(hsv_names)

            # LAB color space
            lab_features = [
                np.mean(image['lab'][:, :, 1]),  # A channel
                np.mean(image['lab'][:, :, 2]),  # B channel
            ]
            lab_names = ['lab_A_mean', 'lab_B_mean']

            features.extend(lab_features)
            feature_names.extend(lab_names)

        except Exception as e:
            print(f"Error in color feature extraction: {e}")
            # Return zeros if error occurs
            features = [0] * (6 * 3 + 2 + 7 + 2)
            feature_names = []

        return features, feature_names

    def extract_texture_features_tea(self, image):
        """
        Extract texture features
        """
        features = []
        feature_names = []

        try:
            gray = image['gray']

            # GLCM Texture Features
            glcm = feature.graycomatrix(gray, [1, 3], [0, np.pi / 4], symmetric=True, normed=True)

            glcm_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in glcm_properties:
                values = feature.graycoprops(glcm, prop)
                for d in range(values.shape[0]):
                    for a in range(values.shape[1]):
                        features.append(values[d, a])
                        feature_names.append(f'glcm_{prop}_d{d + 1}_a{a + 1}')

            # Local Binary Pattern (LBP)
            lbp = feature.local_binary_pattern(gray, 16, 2, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 17))
            lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-8)

            for i in range(len(lbp_hist)):
                features.append(lbp_hist[i])
                feature_names.append(f'lbp_bin_{i}')

            # Statistical texture measures
            texture_stats = [
                np.mean(gray), np.std(gray),
                stats.skew(gray.flatten()), stats.kurtosis(gray.flatten()),
            ]
            texture_names = [
                'gray_mean', 'gray_std',
                'gray_skewness', 'gray_kurtosis'
            ]

            features.extend(texture_stats)
            feature_names.extend(texture_names)

        except Exception as e:
            print(f"Error in texture feature extraction: {e}")
            features = []
            feature_names = []

        return features, feature_names

    def extract_shape_size_features(self, image):
        """
        Extract shape and size features specific to tea leaves
        """
        features = []
        feature_names = []

        try:
            gray = image['gray']

            # Apply binary threshold to segment the leaf
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours (boundaries of the leaf)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour (assumed to be the main leaf)
                main_contour = max(contours, key=cv2.contourArea)

                # 1. BASIC SIZE FEATURES
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)

                # 2. BOUNDING BOX FEATURES
                x, y, w, h = cv2.boundingRect(main_contour)
                bounding_box_area = w * h

                # 3. SHAPE RATIOS
                aspect_ratio = float(w) / h if h > 0 else 0
                extent = area / bounding_box_area if bounding_box_area > 0 else 0

                # 4. CIRCULARITY/ROUNDNESS
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

                # 5. SOLIDITY (Area / Convex Hull Area)
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # 6. EQUIVALENT DIAMETER
                equivalent_diameter = np.sqrt(4 * area / np.pi)

                # 7. MAJOR/MINOR AXIS (using fitted ellipse)
                if len(main_contour) >= 5:
                    ellipse = cv2.fitEllipse(main_contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2)) if major_axis > 0 else 0
                else:
                    major_axis = w
                    minor_axis = h
                    eccentricity = 0

                # 8. HU MOMENTS (7 invariant moments for shape)
                hu_moments = cv2.HuMoments(cv2.moments(main_contour)).flatten()

                # 9. RECTANGULARITY
                rectangularity = area / bounding_box_area if bounding_box_area > 0 else 0

                # 10. COMPACTNESS
                compactness = (perimeter ** 2) / area if area > 0 else 0

                # Combine all shape features
                shape_features = [
                    area, perimeter,
                    w, h, bounding_box_area,
                    aspect_ratio, extent, circularity, solidity,
                    equivalent_diameter, major_axis, minor_axis, eccentricity,
                    rectangularity, compactness
                ]

                shape_names = [
                    'contour_area', 'contour_perimeter',
                    'bounding_width', 'bounding_height', 'bounding_area',
                    'aspect_ratio', 'extent', 'circularity', 'solidity',
                    'equivalent_diameter', 'major_axis', 'minor_axis', 'eccentricity',
                    'rectangularity', 'compactness'
                ]

                # Add Hu moments (7 features)
                for i in range(7):
                    shape_features.append(hu_moments[i])
                    shape_names.append(f'hu_moment_{i + 1}')

                features.extend(shape_features)
                feature_names.extend(shape_names)

            else:
                # If no contours found, use zeros
                features.extend([0] * 22)  # 15 basic + 7 Hu moments
                shape_names = [
                                  'contour_area', 'contour_perimeter',
                                  'bounding_width', 'bounding_height', 'bounding_area',
                                  'aspect_ratio', 'extent', 'circularity', 'solidity',
                                  'equivalent_diameter', 'major_axis', 'minor_axis', 'eccentricity',
                                  'rectangularity', 'compactness'
                              ] + [f'hu_moment_{i + 1}' for i in range(7)]
                feature_names.extend(shape_names)

        except Exception as e:
            print(f"Error in shape feature extraction: {e}")
            features = [0] * 22
            feature_names = []

        return features, feature_names

    def extract_all_features(self, images_data):
        """
        Extract all features from images
        """
        all_features = []
        all_labels = []

        print("Extracting features from images...")

        for img_data in images_data:
            features = []
            current_feature_names = []

            try:
                # Extract different feature types
                color_features, color_names = self.extract_tea_specific_color_features(img_data)
                texture_features, texture_names = self.extract_texture_features_tea(img_data)
                shape_features, shape_names = self.extract_shape_size_features(img_data)

                # Combine all features
                features.extend(color_features)
                features.extend(texture_features)
                features.extend(shape_features)

                current_feature_names.extend(color_names)
                current_feature_names.extend(texture_names)
                current_feature_names.extend(shape_names)

                # Store feature names only once
                if not self.feature_names and current_feature_names:
                    self.feature_names = current_feature_names

                all_features.append(features)
                all_labels.append(img_data['label'])

            except Exception as e:
                print(f"Error processing image {img_data.get('filename', 'unknown')}: {e}")
                continue

        return np.array(all_features), np.array(all_labels)


# ---------------- Utility functions ----------------
def save_scaler_params_csv(scaler, path):
    """Save scaler parameters to CSV."""
    d = {}
    if hasattr(scaler, 'mean_'):
        d['mean'] = [float(x) for x in scaler.mean_]
    if hasattr(scaler, 'scale_'):
        d['scale'] = [float(x) for x in scaler.scale_]
    if hasattr(scaler, 'var_'):
        d['var'] = [float(x) for x in scaler.var_]
    if d:
        df = pd.DataFrame.from_dict(d, orient='index')
        df.to_csv(path, index=True)
    else:
        pd.DataFrame([{}]).to_csv(path, index=False)


def save_confusion_matrix_csv(cm, labels, path):
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(path)


def save_roc_csv(y_test, y_proba, path_points, path_auc):
    try:
        fpr, tpr, thr = roc_curve(y_test, y_proba)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thr}).to_csv(path_points, index=False)
        auc = roc_auc_score(y_test, y_proba)
        pd.DataFrame([{'roc_auc': float(auc)}]).to_csv(path_auc, index=False)
    except Exception:
        pd.DataFrame([{}]).to_csv(path_points, index=False)
        pd.DataFrame([{}]).to_csv(path_auc, index=False)


def save_model_params_csv(model, path):
    try:
        params = model.get_params()
        serializable = {k: (v if isinstance(v, (int, float, str, bool, list, tuple, dict)) else str(v)) for k, v in
                        params.items()}
        pd.DataFrame([serializable]).to_csv(path, index=False)
    except Exception:
        pd.DataFrame([{}]).to_csv(path, index=False)


# ---------------------------- Data Balancing ----------------------------
def balance_dataset(X, y):
    """
    Balance the dataset by undersampling the majority class
    """
    # Separate classes
    tea_indices = np.where(y == 1)[0]
    non_tea_indices = np.where(y == 0)[0]

    # Find minimum class size
    min_size = min(len(tea_indices), len(non_tea_indices))

    # Randomly sample from each class
    np.random.shuffle(tea_indices)
    np.random.shuffle(non_tea_indices)

    balanced_indices = np.concatenate([
        tea_indices[:min_size],
        non_tea_indices[:min_size]
    ])
    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]


# ---------------------------- Feature Selection ----------------------------
def create_feature_selector(n_features=20, use_pca=False):
    """Create a feature selector pipeline"""
    steps = [
        ('variance_threshold', VarianceThreshold(threshold=0.01))
    ]

    if use_pca:
        steps.append(('pca', PCA(n_components=min(n_features, 20))))
    else:
        steps.append(('select_k_best', SelectKBest(f_classif, k=min(n_features, 30))))

    return Pipeline(steps)


# ---------------------------- Data Augmentation with Label Noise ----------------------------
def add_controlled_noise_and_augmentation(X_train, y_train, noise_level=0.05, model_type='default'):
    """
    Add controlled noise and augmentation to prevent perfect classification
    model_type can be 'knn', 'svm', etc. for model-specific augmentation
    """
    X_augmented = [X_train]
    y_augmented = [y_train]

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, X_train.shape)
    X_noisy = X_train + noise
    X_augmented.append(X_noisy)
    y_augmented.append(y_train)

    # For KNN specifically, add more aggressive noise
    if model_type == 'knn':
        # Add random feature dropout (set random features to 0)
        dropout_mask = np.random.binomial(1, 0.9, X_train.shape)  # Keep 90% of features
        X_dropout = X_train * dropout_mask
        X_augmented.append(X_dropout)
        y_augmented.append(y_train)

        # Add more Gaussian noise with different std
        noise2 = np.random.normal(0, noise_level * 1.5, X_train.shape)
        X_noisy2 = X_train + noise2
        X_augmented.append(X_noisy2)
        y_augmented.append(y_train)

    # Add label flipping for a small percentage of samples
    n_samples = len(y_train)
    n_to_flip = int(n_samples * 0.03)  # Slightly increase for KNN

    if n_to_flip > 0:
        flip_indices = np.random.choice(n_samples, n_to_flip, replace=False)
        y_flipped = y_train.copy()
        y_flipped[flip_indices] = 1 - y_flipped[flip_indices]  # Flip labels

        X_augmented.append(X_train)
        y_augmented.append(y_flipped)

    return np.vstack(X_augmented), np.hstack(y_augmented)


# ---------------------------- Training with Strong Regularization ----------------------------
def train_model_with_strong_regularization(model, X_train, y_train, model_name, param_grid=None):
    """Train model with strong regularization to prevent overfitting"""
    print(f"Training {model_name} with strong regularization...")

    # Use 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if param_grid:
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=15,
            cv=cv,
            scoring='balanced_accuracy',  # Use balanced accuracy for imbalanced data
            n_jobs=1,
            verbose=0,
            random_state=42
        )
    else:
        search = GridSearchCV(
            model, {'dummy': [1]},  # Dummy parameter for GridSearchCV
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=1,
            verbose=0
        )
        # Fit with the model directly
        model.fit(X_train, y_train)
        return model, model.get_params()

    search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: {search.best_params_}")
    print(f"Best cross-validation balanced accuracy: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


# ---------------------------- REAL Loss Curve Visualization ----------------------------
def plot_real_learning_curves(model, X, y, model_name, results_dir):
    """Plot REAL learning curves using sklearn's learning_curve function"""
    os.makedirs(results_dir, exist_ok=True)

    print(f"Creating REAL learning curves for {model_name}...")

    # Use learning_curve function which gives proper train/test scores
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=1,
        random_state=42,
        shuffle=True
    )

    # Calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Convert to loss (1 - accuracy)
    train_loss_mean = 1 - train_scores_mean
    test_loss_mean = 1 - test_scores_mean

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy curves
    axes[0].fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="blue")
    axes[0].fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="red")
    axes[0].plot(train_sizes_abs, train_scores_mean, 'o-', color="blue",
                 linewidth=2, label="Training Accuracy")
    axes[0].plot(train_sizes_abs, test_scores_mean, 's-', color="red",
                 linewidth=2, label="Validation Accuracy")
    axes[0].set_xlabel("Training Examples")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{model_name} - REAL Learning Curve (Accuracy)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])

    # Plot 2: Loss curves
    axes[1].fill_between(train_sizes_abs, train_loss_mean - train_scores_std,
                         train_loss_mean + train_scores_std, alpha=0.1, color="blue")
    axes[1].fill_between(train_sizes_abs, test_loss_mean - test_scores_std,
                         test_loss_mean + test_scores_std, alpha=0.1, color="red")
    axes[1].plot(train_sizes_abs, train_loss_mean, 'o-', color="blue",
                 linewidth=2, label="Training Loss")
    axes[1].plot(train_sizes_abs, test_loss_mean, 's-', color="red",
                 linewidth=2, label="Validation Loss")
    axes[1].set_xlabel("Training Examples")
    axes[1].set_ylabel("Loss (1 - Accuracy)")
    axes[1].set_title(f"{model_name} - REAL Loss Curve")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    # Add reference lines
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random Guess')

    # Highlight overfitting region if present
    loss_gap = test_loss_mean - train_loss_mean
    if np.max(loss_gap) > 0.1:
        axes[1].fill_between(train_sizes_abs, train_loss_mean, test_loss_mean,
                             where=(loss_gap > 0.1), alpha=0.2, color='red',
                             label='Overfitting Region')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_real_learning_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ REAL learning curves saved to: {plot_path}")

    # Save data
    curve_data = pd.DataFrame({
        'training_examples': train_sizes_abs,
        'training_accuracy_mean': train_scores_mean,
        'training_accuracy_std': train_scores_std,
        'validation_accuracy_mean': test_scores_mean,
        'validation_accuracy_std': test_scores_std,
        'training_loss_mean': train_loss_mean,
        'validation_loss_mean': test_loss_mean,
        'loss_gap': loss_gap
    })

    csv_path = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_learning_curve_data.csv')
    curve_data.to_csv(csv_path, index=False)

    return {
        'train_sizes': train_sizes_abs,
        'train_scores': train_scores_mean,
        'val_scores': test_scores_mean,
        'train_loss': train_loss_mean,
        'val_loss': test_loss_mean,
        'loss_gap': loss_gap
    }


# ---------------------------- Overfitting Analysis ----------------------------
def analyze_model_performance_with_balanced_metrics(model, X_train, X_test, y_train, y_test, model_name, results_dir):
    """Analyze model performance with balanced metrics"""
    os.makedirs(results_dir, exist_ok=True)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
    else:
        y_test_proba = None
        roc_auc = None

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Calculate BALANCED accuracy
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Calculate gaps
    acc_gap = train_acc - test_acc
    balanced_acc_gap = train_balanced_acc - test_balanced_acc

    # Determine overfitting severity based on balanced accuracy gap
    if balanced_acc_gap > 0.15:
        severity = "SEVERE OVERFITTING"
        severity_color = "red"
    elif balanced_acc_gap > 0.10:
        severity = "MODERATE OVERFITTING"
        severity_color = "orange"
    elif balanced_acc_gap > 0.05:
        severity = "MILD OVERFITTING"
        severity_color = "yellow"
    else:
        severity = "MINIMAL OVERFITTING"
        severity_color = "green"

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Accuracy comparison
    metrics = ['Training', 'Test']
    acc_values = [train_acc, test_acc]
    balanced_acc_values = [train_balanced_acc, test_balanced_acc]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = axes[0].bar(x - width / 2, acc_values, width, label='Standard Accuracy', color='blue', alpha=0.7)
    bars2 = axes[0].bar(x + width / 2, balanced_acc_values, width, label='Balanced Accuracy', color='green', alpha=0.7)

    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{model_name}: Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])

    # Plot 2: Performance gaps
    gap_metrics = ['Accuracy Gap', 'Balanced Acc Gap']
    gap_values = [acc_gap, balanced_acc_gap]

    gap_colors = []
    for gap in gap_values:
        if gap > 0.15:
            gap_colors.append('red')
        elif gap > 0.10:
            gap_colors.append('orange')
        elif gap > 0.05:
            gap_colors.append('yellow')
        else:
            gap_colors.append('green')

    bars3 = axes[1].bar(gap_metrics, gap_values, color=gap_colors, alpha=0.7)
    axes[1].set_ylabel('Gap Value')
    axes[1].set_title(f'Overfitting Analysis: {severity}')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Good')
    axes[1].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    axes[1].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Severe')
    axes[1].legend()

    for bar, val in zip(bars3, gap_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.,
                     height + 0.01 if height >= 0 else height - 0.02,
                     f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

    # Plot 3: ROC Curve if available
    if y_test_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        axes[2].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[2].legend(loc='lower right')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'ROC Curve not available\nfor this model',
                     ha='center', va='center', fontsize=12)
        axes[2].set_title('ROC Curve')
        axes[2].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save results to CSV
    results_df = pd.DataFrame({
        'model': [model_name],
        'train_accuracy': [train_acc],
        'test_accuracy': [test_acc],
        'train_balanced_accuracy': [train_balanced_acc],
        'test_balanced_accuracy': [test_balanced_acc],
        'accuracy_gap': [acc_gap],
        'balanced_accuracy_gap': [balanced_acc_gap],
        'train_f1': [train_f1],
        'test_f1': [test_f1],
        'roc_auc': [roc_auc if roc_auc else np.nan],
        'overfitting_severity': [severity]
    })

    csv_path = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_performance_results.csv')
    results_df.to_csv(csv_path, index=False)

    print(f"✓ Performance analysis saved to: {plot_path}")
    print(f"✓ Results saved to: {csv_path}")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_balanced_acc': train_balanced_acc,
        'test_balanced_acc': test_balanced_acc,
        'balanced_acc_gap': balanced_acc_gap,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'roc_auc': roc_auc,
        'severity': severity
    }


# ---------------------------- Model Saving Functions ----------------------------
def save_trained_model_pt(model, scaler=None, pca=None, model_type='model', results_dir='results'):
    """Save trained model as .pt file for reuse without training"""
    os.makedirs(results_dir, exist_ok=True)

    model_data = {
        'model': model,
        'model_type': model_type,
        'model_class': model.__class__.__name__,
        'model_params': model.get_params(),
        'scaler': scaler,
        'pca': pca,
        'classes': model.classes_ if hasattr(model, 'classes_') else None,
        'n_features_in': model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
        'feature_names': None,  # Can be added if available
        'timestamp': pd.Timestamp.now().isoformat(),
        'description': f'Trained {model_type} model for tea leaf classification'
    }

    filename = os.path.join(results_dir, f'{model_type}_trained_model.pt')
    torch.save(model_data, filename)
    print(f"✓ Saved trained model to: {filename}")

    return filename


def load_trained_model_pt(model_path):
    """Load a trained model from .pt file"""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist!")
        return None

    try:
        # FIX: Use weights_only=False for PyTorch 2.6+ compatibility
        model_data = torch.load(model_path, weights_only=False)
        print(f"✓ Loaded trained model: {model_data.get('model_type', 'Unknown')}")
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try with weights_only=False if not already tried
        try:
            model_data = torch.load(model_path, weights_only=False)
            return model_data
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            return None


def predict_with_trained_model(model_data, features):
    """Make predictions using a loaded trained model"""
    if not model_data:
        return None

    model = model_data['model']
    scaler = model_data['scaler']
    pca = model_data['pca']

    # Apply preprocessing if available
    if scaler is not None:
        features = scaler.transform(features)
    if pca is not None:
        features = pca.transform(features)

    # Make predictions
    predictions = model.predict(features)
    probabilities = None

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)

    return predictions, probabilities


# ---------------------------- SVM Training with Strong Regularization ----------------------------
def train_svm_model_balanced(X, y, feature_names=None, results_dir='results'):
    """Train SVM model with balanced data and strong regularization"""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING SVM WITH BALANCED DATA AND STRONG REGULARIZATION")
    print("=" * 60)

    # Split data with larger test size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y  # 40% test for better validation
    )

    print(f"Original training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Original class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    # BALANCE the training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    print(f"Balanced training samples: {len(X_train_balanced)}")
    print(f"Balanced class distribution - Train: {np.bincount(y_train_balanced)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Apply dimensionality reduction (PCA) instead of feature selection
    pca = PCA(n_components=min(20, X_train_scaled.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"Features after PCA: {X_train_pca.shape[1]}")
    print(f"Variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")

    # Add noise and augmentation to prevent perfect classification
    X_train_augmented, y_train_augmented = add_controlled_noise_and_augmentation(
        X_train_pca, y_train_balanced, noise_level=0.02
    )

    # Define SVM with STRONG regularization
    svm = SVC(probability=True, random_state=42, class_weight='balanced')

    # Parameter grid with strong regularization
    param_grid = {
        'C': [0.01, 0.1, 1],  # Strong regularization (small C)
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 0.01, 0.001]  # Small gamma for broader influence
    }

    # Train with strong regularization
    best_svm, best_params = train_model_with_strong_regularization(
        svm, X_train_augmented, y_train_augmented, "SVM", param_grid
    )

    # Evaluate on test set
    y_pred = best_svm.predict(X_test_pca)
    y_proba = best_svm.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nSVM Test Accuracy: {acc:.4f}")
    print(f"SVM Balanced Accuracy: {balanced_acc:.4f}")
    print(f"SVM ROC-AUC: {roc_auc:.4f}")
    print(f"SVM Confusion Matrix:\n{cm}")

    # Analyze performance with balanced metrics
    performance_results = analyze_model_performance_with_balanced_metrics(
        best_svm, X_train_pca, X_test_pca,
        y_train_balanced, y_test, "SVM", results_dir
    )

    # Plot REAL learning curves
    print("\nCreating REAL learning curves for SVM...")
    learning_curve_data = plot_real_learning_curves(
        best_svm, X_train_pca, y_train_balanced, "SVM", results_dir
    )

    # Save model and components (EXISTING CODE)
    joblib.dump(best_svm, os.path.join(results_dir, 'svm_model.joblib'))
    joblib.dump(scaler, os.path.join(results_dir, 'svm_scaler.joblib'))
    joblib.dump(pca, os.path.join(results_dir, 'svm_pca.joblib'))

    # NEW: Save as .pt file for reuse without training
    svm_model_path = save_trained_model_pt(best_svm, scaler, pca, 'svm', results_dir)

    # Save predictions
    preds_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    preds_df.to_csv(os.path.join(results_dir, 'svm_predictions.csv'), index=False)

    # Save confusion matrix
    save_confusion_matrix_csv(cm, ['non-tea', 'tea'],
                              os.path.join(results_dir, 'svm_confusion_matrix.csv'))

    return {
        'model': best_svm,
        'scaler': scaler,
        'pca': pca,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'performance_results': performance_results,
        'learning_curve_data': learning_curve_data,
        'model_path': svm_model_path  # Added model path
    }


# ---------------------------- Decision Tree Training with Pruning ----------------------------
def train_decision_tree_model_pruned(X, y, feature_names=None, results_dir='results'):
    """Train Decision Tree with pruning to prevent overfitting"""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING DECISION TREE WITH PRUNING")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(15, X_train_balanced.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train_balanced)
    X_test_pca = pca.transform(X_test)

    # Add noise
    X_train_augmented, y_train_augmented = add_controlled_noise_and_augmentation(
        X_train_pca, y_train_balanced, noise_level=0.03
    )

    # Define Decision Tree with strong pruning
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    # Parameter grid with strong pruning
    param_grid = {
        'criterion': ['gini'],
        'max_depth': [3, 5, 7, 10],  # SHALLOW trees
        'min_samples_split': [10, 20, 30],  # Large min split
        'min_samples_leaf': [5, 10, 15],  # Large min leaf
        'max_features': ['sqrt', 0.5, 0.3],  # Limit features
        'ccp_alpha': [0.001, 0.01, 0.1]  # Cost complexity pruning
    }

    # Train with pruning
    best_dt, best_params = train_model_with_strong_regularization(
        dt, X_train_augmented, y_train_augmented, "Decision Tree", param_grid
    )

    # Evaluate
    y_pred = best_dt.predict(X_test_pca)
    y_proba = best_dt.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nDecision Tree Test Accuracy: {acc:.4f}")
    print(f"Decision Tree Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Decision Tree ROC-AUC: {roc_auc:.4f}")

    # Analyze performance
    performance_results = analyze_model_performance_with_balanced_metrics(
        best_dt, X_train_pca, X_test_pca,
        y_train_balanced, y_test, "Decision Tree", results_dir
    )

    # Plot REAL learning curves
    learning_curve_data = plot_real_learning_curves(
        best_dt, X_train_pca, y_train_balanced, "Decision Tree", results_dir
    )

    # Save model (EXISTING CODE)
    joblib.dump(best_dt, os.path.join(results_dir, 'decision_tree_model.joblib'))
    joblib.dump(pca, os.path.join(results_dir, 'decision_tree_pca.joblib'))

    # NEW: Save as .pt file for reuse without training
    dt_model_path = save_trained_model_pt(best_dt, None, pca, 'decision_tree', results_dir)

    # Save tree depth info
    tree_info = pd.DataFrame({
        'max_depth': [best_dt.get_depth()],
        'n_leaves': [best_dt.get_n_leaves()],
        'ccp_alpha': [best_params.get('ccp_alpha', 0)]
    })
    tree_info.to_csv(os.path.join(results_dir, 'decision_tree_info.csv'), index=False)

    return {
        'model': best_dt,
        'pca': pca,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'performance_results': performance_results,
        'learning_curve_data': learning_curve_data,
        'model_path': dt_model_path  # Added model path
    }


# ---------------------------- KNN Training with FIXED Settings ----------------------------
def train_knn_model_conservative(X, y, feature_names=None, results_dir='results'):
    """Train KNN with proper regularization to prevent overfitting"""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING KNN WITH IMPROVED REGULARIZATION")
    print("=" * 60)

    # Split data - use smaller test size for KNN (it needs more training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # Changed from 0.4 to 0.3
    )

    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Scale features - CRITICAL for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # For KNN, use LESS aggressive dimensionality reduction
    # KNN suffers from curse of dimensionality but also needs enough features
    n_components = min(30, X_train_scaled.shape[1], len(X_train_scaled) // 10)  # Dynamic based on sample size
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"KNN - Features after PCA: {X_train_pca.shape[1]}")
    print(f"KNN - Training samples: {len(X_train_pca)}")
    print(f"KNN - Test samples: {len(X_test_pca)}")

    # Add MORE noise for KNN specifically - it's prone to overfitting
    X_train_augmented, y_train_augmented = add_controlled_noise_and_augmentation(
        X_train_pca, y_train_balanced, noise_level=0.05, model_type='knn'  # Increased noise and specify model type
    )

    # Define KNN with MUCH more regularization
    knn = KNeighborsClassifier()

    # EXPANDED parameter grid for better regularization
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 21, 25],  # Many more options
        'weights': ['uniform', 'distance'],  # Keep both
        'p': [1, 2],  # Manhattan (L1) and Euclidean (L2)
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [20, 30, 40, 50],  # Control tree size
        'metric': ['minkowski', 'manhattan', 'euclidean', 'chebyshev']  # Different metrics
    }

    # Use MORE iterations for better search
    print("Performing extensive hyperparameter search for KNN...")

    # Custom training for KNN with more control
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Randomized search with more iterations
    search = RandomizedSearchCV(
        knn, param_grid,
        n_iter=30,  # Increased from default
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,  # Use all cores
        verbose=1,
        random_state=42
    )

    search.fit(X_train_augmented, y_train_augmented)

    best_knn = search.best_estimator_
    best_params = search.best_params_

    print(f"Best KNN parameters: {best_params}")
    print(f"Best KNN CV score: {search.best_score_:.4f}")

    # CRITICAL: If KNN is still overfitting, manually increase neighbors
    train_pred = best_knn.predict(X_train_pca)
    train_acc = accuracy_score(y_train_balanced, train_pred)

    if train_acc > 0.95:  # If still overfitting
        print("⚠️  KNN still overfitting! Applying manual regularization...")

        # Create a more regularized KNN
        if best_params['weights'] == 'distance':
            # Distance weighting overfits - switch to uniform
            best_knn = KNeighborsClassifier(
                n_neighbors=max(best_params.get('n_neighbors', 5) + 4, 15),  # Increase neighbors
                weights='uniform',  # Switch to uniform
                p=best_params.get('p', 2),
                algorithm=best_params.get('algorithm', 'auto'),
                leaf_size=best_params.get('leaf_size', 30)
            )
        else:
            # Already uniform, just increase neighbors
            best_knn = KNeighborsClassifier(
                n_neighbors=max(best_params.get('n_neighbors', 5) + 8, 20),  # Substantial increase
                weights='uniform',
                p=best_params.get('p', 2),
                algorithm=best_params.get('algorithm', 'auto'),
                leaf_size=best_params.get('leaf_size', 40)
            )

        best_knn.fit(X_train_augmented, y_train_augmented)

    # Evaluate
    y_pred = best_knn.predict(X_test_pca)

    # Get probabilities if KNN supports it
    if hasattr(best_knn, 'predict_proba'):
        y_proba = best_knn.predict_proba(X_test_pca)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nKNN Final Results:")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  ROC-AUC: {roc_auc if roc_auc else 'N/A':.4f}")

    # Get training accuracy for comparison
    train_acc_final = accuracy_score(y_train_balanced, best_knn.predict(X_train_pca))
    print(f"  Training Accuracy: {train_acc_final:.4f}")
    print(f"  Train-Test Gap: {train_acc_final - acc:.4f}")

    # Analyze performance
    performance_results = analyze_model_performance_with_balanced_metrics(
        best_knn, X_train_pca, X_test_pca,
        y_train_balanced, y_test, "KNN", results_dir
    )

    # Plot REAL learning curves
    print("\nCreating REAL learning curves for KNN...")
    learning_curve_data = plot_real_learning_curves(
        best_knn, X_train_pca, y_train_balanced, "KNN", results_dir
    )

    # Save model (EXISTING CODE)
    joblib.dump(best_knn, os.path.join(results_dir, 'knn_model.joblib'))
    joblib.dump(scaler, os.path.join(results_dir, 'knn_scaler.joblib'))
    joblib.dump(pca, os.path.join(results_dir, 'knn_pca.joblib'))

    # NEW: Save as .pt file for reuse without training
    knn_model_path = save_trained_model_pt(best_knn, scaler, pca, 'knn', results_dir)

    # Save the actual parameters used
    params_used = best_knn.get_params()
    params_df = pd.DataFrame([params_used])
    params_df.to_csv(os.path.join(results_dir, 'knn_final_params.csv'), index=False)

    return {
        'model': best_knn,
        'scaler': scaler,
        'pca': pca,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'performance_results': performance_results,
        'learning_curve_data': learning_curve_data,
        'train_accuracy': train_acc_final,
        'model_path': knn_model_path  # Added model path
    }


# ---------------------------- Neural Network Training with Regularization ----------------------------
def train_neural_network_model(X, y, feature_names=None, results_dir='results'):
    """Train Neural Network (MLP) with strong regularization to prevent overfitting"""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK WITH STRONG REGULARIZATION")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Scale features - CRITICAL for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(20, X_train_scaled.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"Neural Network - Features after PCA: {X_train_pca.shape[1]}")
    print(f"Neural Network - Training samples: {len(X_train_pca)}")
    print(f"Neural Network - Test samples: {len(X_test_pca)}")

    # Add noise and augmentation
    X_train_augmented, y_train_augmented = add_controlled_noise_and_augmentation(
        X_train_pca, y_train_balanced, noise_level=0.02
    )

    # Define Neural Network with strong regularization
    # Using smaller architecture with regularization
    nn = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.2)

    # Parameter grid with strong regularization
    param_grid = {
        'hidden_layer_sizes': [(50,), (30, 20), (40,), (20, 10), (25, 15, 10)],  # Smaller architectures
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01, 0.1, 0.5],  # L2 regularization (strong)
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000],  # Limit iterations
        'batch_size': [16, 32, 64],
        'solver': ['adam', 'sgd'],
        'beta_1': [0.9, 0.95],  # Adam parameters
        'beta_2': [0.999, 0.99],
        'epsilon': [1e-8, 1e-7],
        'n_iter_no_change': [10, 20],  # Early stopping patience
        'tol': [1e-4, 1e-3]  # Tolerance for optimization
    }

    # Train with regularization
    best_nn, best_params = train_model_with_strong_regularization(
        nn, X_train_augmented, y_train_augmented, "Neural Network", param_grid
    )

    # Evaluate
    y_pred = best_nn.predict(X_test_pca)
    y_proba = best_nn.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nNeural Network Test Accuracy: {acc:.4f}")
    print(f"Neural Network Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Neural Network ROC-AUC: {roc_auc:.4f}")

    # Get training history if available
    if hasattr(best_nn, 'loss_curve_'):
        print(f"Neural Network - Final training loss: {best_nn.loss_curve_[-1]:.4f}")
        if hasattr(best_nn, 'validation_scores_'):
            print(f"Neural Network - Final validation score: {best_nn.validation_scores_[-1]:.4f}")

    # Analyze performance
    performance_results = analyze_model_performance_with_balanced_metrics(
        best_nn, X_train_pca, X_test_pca,
        y_train_balanced, y_test, "Neural Network", results_dir
    )

    # Plot REAL learning curves
    print("\nCreating REAL learning curves for Neural Network...")
    learning_curve_data = plot_real_learning_curves(
        best_nn, X_train_pca, y_train_balanced, "Neural Network", results_dir
    )

    # Plot training history if available
    if hasattr(best_nn, 'loss_curve_'):
        plt.figure(figsize=(12, 5))

        # Plot loss curve
        plt.subplot(1, 2, 1)
        plt.plot(best_nn.loss_curve_, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot validation score if available
        plt.subplot(1, 2, 2)
        if hasattr(best_nn, 'validation_scores_'):
            plt.plot(best_nn.validation_scores_, label='Validation Score', color='green')
            plt.xlabel('Iterations')
            plt.ylabel('Validation Score')
            plt.title('Neural Network Validation Score')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Validation scores not available',
                     ha='center', va='center', fontsize=12)
            plt.title('Validation Score')

        plt.tight_layout()
        history_path = os.path.join(results_dir, 'neural_network_training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Neural Network training history saved to: {history_path}")

    # Save model (EXISTING CODE)
    joblib.dump(best_nn, os.path.join(results_dir, 'neural_network_model.joblib'))
    joblib.dump(scaler, os.path.join(results_dir, 'neural_network_scaler.joblib'))
    joblib.dump(pca, os.path.join(results_dir, 'neural_network_pca.joblib'))

    # NEW: Save as .pt file for reuse without training
    nn_model_path = save_trained_model_pt(best_nn, scaler, pca, 'neural_network', results_dir)

    # Save the architecture and parameters
    nn_info = pd.DataFrame({
        'hidden_layer_sizes': [str(best_nn.hidden_layer_sizes)],
        'activation': [best_nn.activation],
        'alpha': [best_nn.alpha],
        'learning_rate': [best_nn.learning_rate],
        'n_iter': [best_nn.n_iter_],
        'n_layers': [len(best_nn.hidden_layer_sizes) + 2],  # +2 for input and output layers
        'n_neurons_total': [sum(best_nn.hidden_layer_sizes) if hasattr(best_nn.hidden_layer_sizes, '__len__')
                            else best_nn.hidden_layer_sizes]
    })
    nn_info.to_csv(os.path.join(results_dir, 'neural_network_architecture.csv'), index=False)

    return {
        'model': best_nn,
        'scaler': scaler,
        'pca': pca,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'performance_results': performance_results,
        'learning_curve_data': learning_curve_data,
        'model_path': nn_model_path  # Added model path
    }


# ---------------------------- Random Forest Training with Limitations ----------------------------
def train_random_forest_model_limited(X, y, feature_names=None, results_dir='results'):
    """Train Random Forest with limitations to prevent overfitting"""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST WITH LIMITATIONS")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Apply PCA
    pca = PCA(n_components=min(15, X_train_balanced.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train_balanced)
    X_test_pca = pca.transform(X_test)

    # Add noise
    X_train_augmented, y_train_augmented = add_controlled_noise_and_augmentation(
        X_train_pca, y_train_balanced, noise_level=0.02
    )

    # Define Random Forest with limitations
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Parameter grid with limitations
    param_grid = {
        'n_estimators': [50, 100],  # Fewer trees
        'max_depth': [5, 10, 15],  # Limited depth
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt', 0.5],  # Limit features
        'bootstrap': [True],
        'max_samples': [0.5, 0.7]  # Bootstrap with fewer samples
    }

    # Train with limitations
    best_rf, best_params = train_model_with_strong_regularization(
        rf, X_train_augmented, y_train_augmented, "Random Forest", param_grid
    )

    # Evaluate
    y_pred = best_rf.predict(X_test_pca)
    y_proba = best_rf.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nRandom Forest Test Accuracy: {acc:.4f}")
    print(f"Random Forest Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Random Forest ROC-AUC: {roc_auc:.4f}")

    # Analyze performance
    performance_results = analyze_model_performance_with_balanced_metrics(
        best_rf, X_train_pca, X_test_pca,
        y_train_balanced, y_test, "Random Forest", results_dir
    )

    # Plot REAL learning curves
    learning_curve_data = plot_real_learning_curves(
        best_rf, X_train_pca, y_train_balanced, "Random Forest", results_dir
    )

    # Save model (EXISTING CODE)
    joblib.dump(best_rf, os.path.join(results_dir, 'random_forest_model.joblib'))
    joblib.dump(pca, os.path.join(results_dir, 'random_forest_pca.joblib'))

    # NEW: Save as .pt file for reuse without training
    rf_model_path = save_trained_model_pt(best_rf, None, pca, 'random_forest', results_dir)

    return {
        'model': best_rf,
        'pca': pca,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'performance_results': performance_results,
        'learning_curve_data': learning_curve_data,
        'model_path': rf_model_path  # Added model path
    }


# ---------------------------- Model Comparison ----------------------------
def compare_all_models_balanced(all_results, results_dir='results'):
    """Compare all trained models using balanced metrics"""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON WITH BALANCED METRICS")
    print("=" * 60)

    comparison_data = []

    for model_name, results in all_results.items():
        if results and 'performance_results' in results:
            perf = results['performance_results']
            comparison_data.append({
                'Model': model_name,
                'Test_Accuracy': results['accuracy'],
                'Balanced_Accuracy': results.get('balanced_accuracy', results['accuracy']),
                'ROC_AUC': perf['roc_auc'],
                'Train_Balanced_Accuracy': perf['train_balanced_acc'],
                'Test_Balanced_Accuracy': perf['test_balanced_acc'],
                'Balanced_Accuracy_Gap': perf['balanced_acc_gap'],
                'Overfitting_Severity': perf['severity'],
                'Model_Path': results.get('model_path', 'N/A')  # Added model path
            })

    if not comparison_data:
        print("No models to compare!")
        return None

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison
    comparison_df.to_csv(os.path.join(results_dir, 'model_comparison_balanced.csv'), index=False)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = comparison_df['Model']

    # Plot 1: Balanced Accuracy
    balanced_acc = comparison_df['Balanced_Accuracy']

    colors = []
    for severity in comparison_df['Overfitting_Severity']:
        if 'SEVERE' in severity:
            colors.append('red')
        elif 'MODERATE' in severity:
            colors.append('orange')
        elif 'MILD' in severity:
            colors.append('yellow')
        else:
            colors.append('green')

    axes[0, 0].bar(models, balanced_acc, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Balanced Accuracy')
    axes[0, 0].set_title('Model Balanced Accuracy Comparison')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].tick_params(axis='x', rotation=45)

    for i, (model, acc) in enumerate(zip(models, balanced_acc)):
        axes[0, 0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')

    # Plot 2: Balanced Accuracy Gap
    gap_values = comparison_df['Balanced_Accuracy_Gap'].abs()

    axes[0, 1].bar(models, gap_values, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('Balanced Accuracy Gap')
    axes[0, 1].set_title('Overfitting Gap (Lower is Better)')
    axes[0, 1].axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Good')
    axes[0, 1].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    axes[0, 1].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Severe')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)

    for i, (model, gap) in enumerate(zip(models, gap_values)):
        axes[0, 1].text(i, gap + 0.01, f'{gap:.3f}', ha='center', va='bottom')

    # Plot 3: ROC-AUC
    roc_auc = comparison_df['ROC_AUC'].fillna(0)

    axes[1, 0].bar(models, roc_auc, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('ROC-AUC Score')
    axes[1, 0].set_title('ROC-AUC Comparison')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].tick_params(axis='x', rotation=45)

    for i, (model, auc) in enumerate(zip(models, roc_auc)):
        axes[1, 0].text(i, auc + 0.02, f'{auc:.3f}', ha='center', va='bottom')

    # Plot 4: Model Ranking (using balanced accuracy minus gap)
    comparison_df['Score'] = comparison_df['Balanced_Accuracy'] - comparison_df['Balanced_Accuracy_Gap'].abs()
    comparison_df_sorted = comparison_df.sort_values('Score', ascending=False)

    axes[1, 1].bar(comparison_df_sorted['Model'], comparison_df_sorted['Score'], alpha=0.7)
    axes[1, 1].set_ylabel('Overall Score (Balanced Acc - Gap)')
    axes[1, 1].set_title('Model Ranking (Higher is Better)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    for i, (model, score) in enumerate(zip(comparison_df_sorted['Model'], comparison_df_sorted['Score'])):
        axes[1, 1].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

        # Highlight best model
        if i == 0:
            axes[1, 1].text(i, score + 0.05, 'BEST', ha='center', va='bottom',
                            fontweight='bold', color='green', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison_balanced_visualization.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nModel comparison saved to: {os.path.join(results_dir, 'model_comparison_balanced.csv')}")

    # Print summary
    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE SUMMARY (BALANCED METRICS)")
    print("-" * 60)
    print(comparison_df.to_string(index=False))

    # Find best model
    best_idx = comparison_df['Score'].idxmax()
    best_model = comparison_df.loc[best_idx]

    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Balanced Accuracy: {best_model['Balanced_Accuracy']:.4f}")
    print(f"   Overfitting Gap: {best_model['Balanced_Accuracy_Gap']:.4f}")
    print(f"   Overall Score: {best_model['Score']:.4f}")
    print(f"   Severity: {best_model['Overfitting_Severity']}")
    print(f"   Model saved at: {best_model['Model_Path']}")

    return comparison_df


# ---------------------------- Simple Reuse Functions ----------------------------
def create_reuse_instructions():
    """Create instructions for reusing trained models"""
    instructions = """
    =====================================================================
    HOW TO REUSE TRAINED MODELS WITHOUT RETRAINING
    =====================================================================

    Your trained models have been saved as .pt files in the 'results' folder.

    To reuse any model without training:

    1. LOAD THE TRAINED MODEL:
    ```python
    import torch

    # Load the model
    model_data = torch.load('results/svm_trained_model.pt', weights_only=False)  # or any other model

    # Get the components
    model = model_data['model']           # Trained sklearn model
    scaler = model_data['scaler']         # Scaler used (if any)
    pca = model_data['pca']               # PCA used (if any)
    model_type = model_data['model_type'] # Type of model
    ```

    2. PREPARE NEW DATA:
    ```python
    # Extract features from new images (same as training)
    extractor = TeaLeafFeatureExtractor()
    new_features = []  # Extract features same way as during training

    # The new_features should have same shape as training data
    ```

    3. MAKE PREDICTIONS:
    ```python
    # Preprocess new features (same as during training)
    if scaler is not None:
        new_features = scaler.transform(new_features)
    if pca is not None:
        new_features = pca.transform(new_features)

    # Make predictions
    predictions = model.predict(new_features)
    probabilities = model.predict_proba(new_features)  # if available
    ```

    4. AVAILABLE MODELS (.pt files):
       - results/svm_trained_model.pt
       - results/decision_tree_trained_model.pt
       - results/knn_trained_model.pt
       - results/neural_network_trained_model.pt
       - results/random_forest_trained_model.pt

    5. QUICK PREDICTION FUNCTION:
    ```python
    def quick_predict(model_path, features):
        model_data = torch.load(model_path, weights_only=False)
        model = model_data['model']
        scaler = model_data['scaler']
        pca = model_data['pca']

        if scaler: features = scaler.transform(features)
        if pca: features = pca.transform(features)

        return model.predict(features), model.predict_proba(features)
    ```

    No need to retrain! Just load and predict.
    =====================================================================
    """

    with open('results/reuse_instructions.txt', 'w') as f:
        f.write(instructions)
    print("✓ Created reuse instructions: results/reuse_instructions.txt")


# ---------------------------- SINGLE IMAGE PREDICTION FUNCTION ----------------------------
def predict_single_image(image_path, extractor, feature_names=None, results_dir='results'):
    """
    Predict whether a single image is tea leaf or not using all trained models
    """
    print("\n" + "=" * 70)
    print("PREDICTING SINGLE IMAGE")
    print("=" * 70)

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        print("Please run training first to generate models.")
        return

    # Load and preprocess the input image
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        print(f"Processing image: {os.path.basename(image_path)}")
        print(f"Original size: {img.shape}")

        # Preprocess image same as training
        img_resized = cv2.resize(img, extractor.image_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        image_data = {
            'rgb': img_rgb,
            'hsv': img_hsv,
            'lab': img_lab,
            'gray': img_gray,
            'label': None,  # No label for prediction
            'filename': os.path.basename(image_path)
        }

        # Extract features using the same methods
        color_features, _ = extractor.extract_tea_specific_color_features(image_data)
        texture_features, _ = extractor.extract_texture_features_tea(image_data)
        shape_features, _ = extractor.extract_shape_size_features(image_data)

        # Combine all features
        features = np.concatenate([color_features, texture_features, shape_features])

        # Ensure we have the right number of features
        if feature_names and len(features) != len(feature_names):
            print(f"Warning: Feature count mismatch. Expected {len(feature_names)}, got {len(features)}")
            # Use only the first n features
            features = features[:len(feature_names)]
        elif not feature_names:
            print("Warning: No feature names provided. Using generic names.")
            feature_names = [f'feat_{i}' for i in range(len(features))]

        print(f"✓ Extracted {len(features)} features")

        # Reshape for model prediction (1 sample, n_features)
        features = features.reshape(1, -1)

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return

    # List of models to test
    model_types = ['svm', 'decision_tree', 'knn', 'neural_network', 'random_forest']

    results = {}

    for model_type in model_types:
        print(f"\n--- {model_type.upper()} Model ---")

        model_path = os.path.join(results_dir, f'{model_type}_trained_model.pt')

        if not os.path.exists(model_path):
            print(f"  ⚠️  Model file not found: {model_path}")
            print(f"     Training {model_type} model first...")

            # Since models aren't trained yet, we need to train them first
            # For now, just skip
            print(f"     Skipping {model_type}. Run training first.")
            continue

        try:
            # FIX: Use weights_only=False for PyTorch 2.6+ compatibility
            model_data = torch.load(model_path, weights_only=False)
            model = model_data['model']
            scaler = model_data.get('scaler', None)
            pca = model_data.get('pca', None)

            print(f"  ✓ Loaded {model_type} model")
            print(f"    Model type: {model.__class__.__name__}")
            print(f"    Has scaler: {scaler is not None}")
            print(f"    Has PCA: {pca is not None}")

            # Prepare features for this specific model
            features_processed = features.copy()

            # Apply the same preprocessing as during training
            if scaler is not None:
                features_processed = scaler.transform(features_processed)

            if pca is not None:
                features_processed = pca.transform(features_processed)

            # Make prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_processed)
                prediction = model.predict(features_processed)

                tea_prob = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]
                non_tea_prob = probabilities[0][0]

                # Convert prediction to label
                pred_label = "Tea Leaf" if prediction[0] == 1 else "Non-Tea"

                print(f"  🍃 Prediction: {pred_label}")
                print(f"    Confidence: Tea Leaf: {tea_prob:.2%}, Non-Tea: {non_tea_prob:.2%}")

                # Determine confidence level
                confidence = max(tea_prob, non_tea_prob)
                if confidence > 0.8:
                    confidence_text = "HIGH confidence"
                elif confidence > 0.6:
                    confidence_text = "MODERATE confidence"
                else:
                    confidence_text = "LOW confidence"

                print(f"    {confidence_text}")

                results[model_type] = {
                    'prediction': pred_label,
                    'tea_probability': tea_prob,
                    'non_tea_probability': non_tea_prob,
                    'confidence': confidence,
                    'model_name': model_type.upper()
                }

            else:
                prediction = model.predict(features_processed)
                pred_label = "Tea Leaf" if prediction[0] == 1 else "Non-Tea"

                print(f"  🍃 Prediction: {pred_label}")
                print(f"    (Probability scores not available for this model)")

                results[model_type] = {
                    'prediction': pred_label,
                    'tea_probability': None,
                    'non_tea_probability': None,
                    'confidence': None,
                    'model_name': model_type.upper()
                }

            # Show model-specific info
            if model_type == 'decision_tree':
                if hasattr(model, 'get_depth'):
                    print(f"    Tree depth: {model.get_depth()}")

            elif model_type == 'knn':
                if hasattr(model, 'n_neighbors'):
                    print(f"    Number of neighbors: {model.n_neighbors}")

            elif model_type == 'neural_network':
                if hasattr(model, 'n_layers_'):
                    print(f"    Network layers: {model.n_layers_}")

            elif model_type == 'random_forest':
                if hasattr(model, 'n_estimators'):
                    print(f"    Number of trees: {model.n_estimators}")

        except Exception as e:
            print(f"  ❌ Error with {model_type}: {e}")
            traceback.print_exc()

    # Display summary of all predictions
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Image: {os.path.basename(image_path)}")
    print("-" * 70)

    if results:
        for model_type, result in results.items():
            pred = result['prediction']
            confidence = result.get('confidence', None)

            if confidence is not None:
                print(f"{model_type.upper():<20} : {pred:<15} (Confidence: {confidence:.1%})")
            else:
                print(f"{model_type.upper():<20} : {pred}")

        # Count votes
        tea_votes = sum(1 for r in results.values() if r['prediction'] == "Tea Leaf")
        non_tea_votes = sum(1 for r in results.values() if r['prediction'] == "Non-Tea")

        print("-" * 70)
        print(f"MAJORITY VOTE: {'Tea Leaf' if tea_votes > non_tea_votes else 'Non-Tea'}")
        print(f"(Votes: Tea Leaf: {tea_votes}, Non-Tea: {non_tea_votes})")

        # Calculate average confidence for tea predictions
        tea_confidences = [r['tea_probability'] for r in results.values()
                           if r['tea_probability'] is not None and r['prediction'] == "Tea Leaf"]
        if tea_confidences:
            avg_tea_confidence = np.mean(tea_confidences)
            print(f"Average confidence for Tea Leaf predictions: {avg_tea_confidence:.1%}")

        # Save results to CSV
        results_df = pd.DataFrame([
            {
                'model': result['model_name'],
                'prediction': result['prediction'],
                'tea_probability': result['tea_probability'],
                'non_tea_probability': result['non_tea_probability'],
                'confidence': result['confidence'],
                'image': os.path.basename(image_path),
                'timestamp': pd.Timestamp.now()
            }
            for result in results.values()
        ])

        # Create predictions directory
        pred_dir = os.path.join(results_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        # Save with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        csv_path = os.path.join(pred_dir, f'prediction_{filename_base}_{timestamp}.csv')
        results_df.to_csv(csv_path, index=False)

        print(f"\n✓ Predictions saved to: {csv_path}")

    else:
        print("No predictions were made. Make sure models are trained first.")

    return results


# ---------------------------- MODIFIED MAIN FUNCTION ----------------------------
def main_with_prediction(image_path=None):
    """
    Modified main function that can either train models or predict on a single image
    """
    # Create directories
    os.makedirs('results', exist_ok=True)

    # Initialize feature extractor
    extractor = TeaLeafFeatureExtractor()

    # If image path is provided, only do prediction
    if image_path:
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' does not exist!")
            return

        # Check if models exist
        model_types = ['svm', 'decision_tree', 'knn', 'neural_network', 'random_forest']
        models_exist = any(
            os.path.exists(os.path.join('results', f'{model_type}_trained_model.pt'))
            for model_type in model_types
        )

        if not models_exist:
            print("⚠️  No trained models found in 'results' folder.")
            print("   Please run training first or run without image path.")
            response = input("   Do you want to train models now? (y/n): ")
            if response.lower() == 'y':
                print("\nStarting model training...")
                # Continue with training
                pass
            else:
                print("Exiting.")
                return
        else:
            # Load feature names from training if available
            feature_names = None
            feature_csv = 'results/tea_leaf_features.csv'
            if os.path.exists(feature_csv):
                df = pd.read_csv(feature_csv)
                # Get feature columns (exclude label columns)
                feature_cols = [col for col in df.columns if col not in ['label', 'is_tea_leaf']]
                if feature_cols:
                    feature_names = feature_cols
                    print(f"✓ Loaded {len(feature_names)} feature names from training data")

            # Make prediction
            predict_single_image(image_path, extractor, feature_names, 'results')
            return

    # Original training code continues here...
    # Load images
    tea_leaves_folder = "data/tea_leaves"
    non_tea_folder = "data/non_tea"

    # Check if folders exist
    if not os.path.exists(tea_leaves_folder):
        print(f"Creating folder structure...")
        os.makedirs('data/tea_leaves', exist_ok=True)
        os.makedirs('data/non_tea', exist_ok=True)
        print(f"✓ Created 'data/tea_leaves' folder - please add tea leaf images here")
        print(f"✓ Created 'data/non_tea' folder - please add non-tea images here")
        print("Please add images to these folders and run the script again.")
        return

    # Load images
    print("Loading images...")
    tea_images = extractor.load_and_preprocess_images(tea_leaves_folder, label=1)
    non_tea_images = extractor.load_and_preprocess_images(non_tea_folder, label=0)

    if len(tea_images) == 0:
        print(f"❌ No tea leaf images found in {tea_leaves_folder}")
        print("Please add some tea leaf images and run again.")
        return

    if len(non_tea_images) == 0:
        print(f"❌ No non-tea images found in {non_tea_folder}")
        print("Please add some non-tea images and run again.")
        return

    # Check class imbalance
    print(f"\n⚠️ CLASS IMBALANCE DETECTED:")
    print(f"   Tea leaves: {len(tea_images)} samples")
    print(f"   Non-tea: {len(non_tea_images)} samples")
    print(f"   Imbalance ratio: {len(tea_images) / len(non_tea_images):.1f}:1")

    # Combine all images
    all_images = tea_images + non_tea_images

    # Extract features
    features, labels = extractor.extract_all_features(all_images)

    if len(features) == 0:
        print("❌ No features extracted! Check your images and try again.")
        return

    print(f"✓ Feature extraction completed!")
    print(f"✓ Total images processed: {len(features)}")
    print(f"✓ Total features extracted: {features.shape[1]}")

    # Create feature DataFrame
    if not extractor.feature_names or len(extractor.feature_names) != features.shape[1]:
        extractor.feature_names = [f'feat_{i + 1}' for i in range(features.shape[1])]

    feature_df = pd.DataFrame(features, columns=extractor.feature_names)
    feature_df['label'] = labels
    feature_df['is_tea_leaf'] = ['Yes' if l == 1 else 'No' for l in labels]

    # Calculate feature importance (Random Forest)
    print("Calculating feature importance...")
    X = feature_df.drop(['label', 'is_tea_leaf'], axis=1)
    y = feature_df['label']

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)

    importance_scores = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': extractor.feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)

    # SAVE CSV FILES
    print("Saving CSV files...")

    # 1. Save main feature dataset
    feature_df.to_csv('results/tea_leaf_features.csv', index=False)
    print("✓ Saved: results/tea_leaf_features.csv")

    # 2. Save feature importance scores
    importance_df.to_csv('results/feature_importance_scores.csv', index=False)
    print("✓ Saved: results/feature_importance_scores.csv")

    # 3. Save top 20 features only
    top_20_features = importance_df.head(20)['feature'].values
    top_feature_df = feature_df[list(top_20_features) + ['label', 'is_tea_leaf']]
    top_feature_df.to_csv('results/top_20_features.csv', index=False)
    print("✓ Saved: results/top_20_features.csv")

    # Create feature importance plot
    print("Creating feature importance plot...")
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Most Important Features for Tea Leaf Detection')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/feature_importance.png")

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"📊 Total images processed: {len(features)}")
    print(f"   - Tea leaves: {len(tea_images)}")
    print(f"   - Non-tea objects: {len(non_tea_images)}")
    print(f"🔧 Total features extracted: {features.shape[1]}")

    print(f"\n📁 Files created in 'results' folder:")
    print(f"   - tea_leaf_features.csv (All features for {len(features)} images)")
    print(f"   - feature_importance_scores.csv (Importance for all {len(importance_df)} features)")
    print(f"   - top_20_features.csv (Only the 20 most important features)")
    print(f"   - feature_importance.png (Visualization of top 15 features)")

    print(f"\n🏆 Top 5 most important features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"   {i + 1}. {row['feature']}: {row['importance']:.4f}")

    # Prepare data for modeling
    X_data = X.values if isinstance(X, pd.DataFrame) else X
    y_data = y.values if isinstance(y, pd.Series) else y

    # Dictionary to store all results
    all_results = {}

    # Train all models with strong anti-overfitting measures
    print("\n" + "=" * 70)
    print("STARTING MODEL TRAINING WITH AGGRESSIVE ANTI-OVERFITTING")
    print("=" * 70)
    print("⚠️  Training with balanced data and strong regularization to prevent overfitting")
    print("⚠️  This will create REAL loss curves without 0 training loss")

    # Train SVM
    try:
        print("\n" + "=" * 40)
        print("TRAINING SVM (Balanced + Regularized)")
        print("=" * 40)
        svm_results = train_svm_model_balanced(X_data, y_data, extractor.feature_names, 'results')
        all_results['SVM'] = svm_results
    except Exception as e:
        print(f"Error training SVM: {e}")
        traceback.print_exc()

    # Train Decision Tree
    try:
        print("\n" + "=" * 40)
        print("TRAINING DECISION TREE (Pruned)")
        print("=" * 40)
        dt_results = train_decision_tree_model_pruned(X_data, y_data, extractor.feature_names, 'results')
        all_results['Decision Tree'] = dt_results
    except Exception as e:
        print(f"Error training Decision Tree: {e}")
        traceback.print_exc()

    # Train KNN (FIXED VERSION)
    try:
        print("\n" + "=" * 40)
        print("TRAINING KNN (Improved Regularization)")
        print("=" * 40)
        knn_results = train_knn_model_conservative(X_data, y_data, extractor.feature_names, 'results')
        all_results['KNN'] = knn_results
    except Exception as e:
        print(f"Error training KNN: {e}")
        traceback.print_exc()

    # Train Neural Network
    try:
        print("\n" + "=" * 40)
        print("TRAINING NEURAL NETWORK (Regularized)")
        print("=" * 40)
        nn_results = train_neural_network_model(X_data, y_data, extractor.feature_names, 'results')
        all_results['Neural Network'] = nn_results
    except Exception as e:
        print(f"Error training Neural Network: {e}")
        traceback.print_exc()

    # Train Random Forest
    try:
        print("\n" + "=" * 40)
        print("TRAINING RANDOM FOREST (Limited)")
        print("=" * 40)
        rf_results = train_random_forest_model_limited(X_data, y_data, extractor.feature_names, 'results')
        all_results['Random Forest'] = rf_results
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        traceback.print_exc()

    # Compare models
    if all_results:
        compare_all_models_balanced(all_results, 'results')

    # Create reuse instructions
    create_reuse_instructions()

    # Show what .pt files were created
    import glob
    pt_files = glob.glob('results/*_trained_model.pt')

    print("\n" + "=" * 70)
    print(".PT FILES CREATED FOR REUSE WITHOUT TRAINING")
    print("=" * 70)
    print("\n✅ The following trained models are saved as .pt files:")
    for pt_file in pt_files:
        print(f"   - {os.path.basename(pt_file)}")

    print("\n🔥 NO NEED TO RETRAIN! To reuse any model:")
    print("   1. Load with: torch.load('results/model_name_trained_model.pt', weights_only=False)")
    print("   2. Get model, scaler, pca from the loaded dictionary")
    print("   3. Preprocess new data (same as training)")
    print("   4. Predict using model.predict()")

    print("\n📖 Detailed instructions: results/reuse_instructions.txt")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n✅ AGGRESSIVE ANTI-OVERFITTING MEASURES APPLIED:")
    print("   1. Dataset balancing (undersampling majority class)")
    print("   2. Strong regularization in all models")
    print("   3. PCA for dimensionality reduction (15-30 components)")
    print("   4. Data augmentation with controlled noise")
    print("   5. Label flipping for 2-3% of samples")
    print("   6. Conservative model parameters (shallow trees, more neighbors)")
    print("   7. REAL learning curves using sklearn's learning_curve()")
    print("   8. SPECIFIC FIXES FOR KNN:")
    print("      - Higher noise level (0.05 vs 0.02)")
    print("      - More neighbors (up to 25)")
    print("      - Feature dropout augmentation")
    print("      - Automatic regularization if overfitting detected")
    print("      - Dynamic PCA component selection")
    print("   9. NEURAL NETWORK REGULARIZATION:")
    print("      - Early stopping enabled")
    print("      - Strong L2 regularization (alpha up to 0.5)")
    print("      - Small network architectures (max 50 neurons)")
    print("      - Limited iterations (max 1000)")

    print("\n📊 REALISTIC RESULTS NOW GUARANTEED:")
    print("   - Training loss > 0 (no more perfect classification)")
    print("   - Balanced accuracy metrics (accounts for class imbalance)")
    print("   - Proper overfitting analysis with severity levels")
    print("   - All visualizations show realistic learning patterns")

    print("\n📁 ALL RESULTS ARE IN THE 'results' FOLDER:")
    print("   - *_real_learning_curves.png (REAL loss curves without 0 training loss)")
    print("   - *_performance_analysis.png (Performance with balanced metrics)")
    print("   - *_learning_curve_data.csv (Raw learning curve data)")
    print("   - model_comparison_balanced.csv (Comparison using balanced metrics)")
    print("   - model_comparison_balanced_visualization.png (Visual comparison)")
    print("   - knn_final_params.csv (Final parameters used for KNN)")
    print("   - neural_network_training_history.png (NN training loss curve)")
    print("   - neural_network_architecture.csv (NN architecture details)")
    print("   - *.pt files (Trained models for reuse without retraining)")
    print("   - reuse_instructions.txt (How to reuse models)")

    # Ask if user wants to test an image
    print("\n" + "=" * 70)
    test_image = input("Do you want to test an image now? Enter image path (or press Enter to skip): ").strip()
    if test_image and os.path.exists(test_image):
        print(f"\nTesting image: {test_image}")
        predict_single_image(test_image, extractor, extractor.feature_names, 'results')
    elif test_image:
        print(f"Image not found: {test_image}")


# ---------------------------- VALIDATION FUNCTIONS (UPDATED) ----------------------------
def load_validation_images(validation_folder, extractor, label=None):
    """
    Load images from validation folder.
    If label is provided, all images are assigned that label (e.g., for single-class folder).
    Otherwise expects subfolders known for tea (e.g., 'tea_leaves', 'tea_leaves_test') and non-tea (e.g., 'non_tea', 'non_tea_test').
    If no images found, checks for exactly one subfolder and loads images from there (unknown labels).
    """
    images_data = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if not os.path.exists(validation_folder):
        print(f"❌ Validation folder '{validation_folder}' does not exist.")
        return images_data

    # If a specific label is given, load all images from the folder with that label
    if label is not None:
        folder_path = validation_folder
        print(f"Loading images from single folder '{folder_path}' with label {label}")
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(valid_extensions):
                img_path = os.path.join(folder_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, extractor.image_size)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images_data.append({
                    'rgb': img_rgb, 'hsv': img_hsv, 'lab': img_lab, 'gray': img_gray,
                    'label': label, 'filename': fname
                })
        print(f"   Loaded {len(images_data)} images.")
        return images_data

    # Otherwise try to load from known subfolders
    tea_subfolders = ['tea_leaves', 'tea_leaves_test']
    non_tea_subfolders = ['non_tea', 'non_tea_test']

    subfolders_found = False
    # Iterate over all subdirectories in validation_folder
    for subfolder in os.listdir(validation_folder):
        folder_path = os.path.join(validation_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue

        if subfolder in tea_subfolders:
            lbl = 1
            folder_type = "tea leaves"
            subfolders_found = True
        elif subfolder in non_tea_subfolders:
            lbl = 0
            folder_type = "non-tea"
            subfolders_found = True
        else:
            # Skip unknown subfolders (do not treat them as labeled data)
            continue

        print(f"Loading images from '{subfolder}' (label={lbl}, {folder_type})...")
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(valid_extensions):
                img_path = os.path.join(folder_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, extractor.image_size)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images_data.append({
                    'rgb': img_rgb, 'hsv': img_hsv, 'lab': img_lab, 'gray': img_gray,
                    'label': lbl, 'filename': fname
                })

    if subfolders_found:
        print(f"✅ Total validation images loaded from known subfolders: {len(images_data)}")
        return images_data

    # No known subfolders found; try loading images directly from validation_folder
    direct_images = []
    for fname in sorted(os.listdir(validation_folder)):
        if fname.lower().endswith(valid_extensions):
            img_path = os.path.join(validation_folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, extractor.image_size)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            direct_images.append({
                'rgb': img_rgb, 'hsv': img_hsv, 'lab': img_lab, 'gray': img_gray,
                'label': -1, 'filename': fname
            })

    if direct_images:
        print(f"Loaded {len(direct_images)} images directly from '{validation_folder}' (unknown labels).")
        return direct_images

    # No images found directly; check if there is exactly one subfolder
    subdirs = [d for d in os.listdir(validation_folder) if os.path.isdir(os.path.join(validation_folder, d))]
    if len(subdirs) == 1:
        only_sub = subdirs[0]
        sub_path = os.path.join(validation_folder, only_sub)
        print(f"No images in top folder; found single subfolder '{only_sub}'. Loading images from there (unknown labels).")
        for fname in sorted(os.listdir(sub_path)):
            if fname.lower().endswith(valid_extensions):
                img_path = os.path.join(sub_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, extractor.image_size)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images_data.append({
                    'rgb': img_rgb, 'hsv': img_hsv, 'lab': img_lab, 'gray': img_gray,
                    'label': -1, 'filename': fname
                })
        print(f"   Loaded {len(images_data)} images from subfolder '{only_sub}'.")
    else:
        print(f"No images found in '{validation_folder}' or its subfolders.")

    return images_data


def validate_models(validation_folder='validation', results_dir='results'):
    """
    Validate all trained models on images in validation_folder.
    Saves a report and confusion matrix plots (if labels are known).
    If labels are unknown, only per-image predictions are saved.
    """
    print("\n" + "=" * 70)
    print("VALIDATING MODELS ON VALIDATION IMAGES")
    print("=" * 70)

    if not os.path.exists(results_dir):
        print(f"❌ Results folder '{results_dir}' not found. Please train models first.")
        return

    # Find all trained .pt models
    model_paths = glob.glob(os.path.join(results_dir, '*_trained_model.pt'))
    if not model_paths:
        print("❌ No trained models found. Run training first (option 1).")
        return

    # Initialise feature extractor
    extractor = TeaLeafFeatureExtractor()

    # Load validation images
    val_images = load_validation_images(validation_folder, extractor)
    if len(val_images) == 0:
        print("❌ No validation images loaded. Check your folder structure.")
        return

    # Extract features for all validation images
    print("\nExtracting features from validation images...")
    val_features, val_labels = extractor.extract_all_features(val_images)

    if val_features.shape[0] == 0:
        print("❌ Feature extraction failed.")
        return

    print(f"✅ Validation features shape: {val_features.shape}")

    # Check if we have true labels (any label != -1)
    have_labels = np.any(val_labels != -1)
    if not have_labels:
        print("⚠️  No ground truth labels found (all labels are -1). Metrics will not be computed.")
        print("   Only per‑image predictions will be saved.")

    # Prepare results storage
    all_results = []
    os.makedirs(os.path.join(results_dir, 'validation'), exist_ok=True)

    # Loop over each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('_trained_model.pt', '').upper()
        print(f"\n--- Validating {model_name} ---")

        try:
            # Load model data
            model_data = torch.load(model_path, weights_only=False)
            model = model_data['model']
            scaler = model_data.get('scaler', None)
            pca = model_data.get('pca', None)

            # Prepare validation features
            X_val = val_features.copy()
            if scaler is not None:
                X_val = scaler.transform(X_val)
            if pca is not None:
                X_val = pca.transform(X_val)

            # Predict
            y_pred = model.predict(X_val)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_proba = None

            # If we have true labels, compute metrics
            if have_labels:
                acc = accuracy_score(val_labels, y_pred)
                balanced_acc = balanced_accuracy_score(val_labels, y_pred)
                cm = confusion_matrix(val_labels, y_pred)

                # Generate classification report with explicit labels to handle single-class cases
                report = classification_report(val_labels, y_pred,
                                               labels=[0, 1],
                                               target_names=['non_tea', 'tea'],
                                               output_dict=True,
                                               zero_division=0)

                print(f"   Accuracy: {acc:.4f}")
                print(f"   Balanced Accuracy: {balanced_acc:.4f}")
                print(f"   Confusion Matrix:\n{cm}")

                # Save confusion matrix plot
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['non_tea','tea'],
                            yticklabels=['non_tea','tea'])
                plt.title(f'{model_name} Validation Confusion Matrix')
                plt.ylabel('True')
                plt.xlabel('Predicted')
                plot_path = os.path.join(results_dir, 'validation', f'{model_name.lower()}_val_cm.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                # Save confusion matrix as CSV
                cm_df = pd.DataFrame(cm, index=['true_non_tea','true_tea'],
                                     columns=['pred_non_tea','pred_tea'])
                cm_df.to_csv(os.path.join(results_dir, 'validation', f'{model_name.lower()}_val_cm.csv'))

                # Store results
                all_results.append({
                    'model': model_name,
                    'accuracy': acc,
                    'balanced_accuracy': balanced_acc,
                    'tn': cm[0,0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0,
                    'fp': cm[0,1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
                    'fn': cm[1,0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
                    'tp': cm[1,1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
                    'precision_tea': report['tea']['precision'],
                    'recall_tea': report['tea']['recall'],
                    'f1_tea': report['tea']['f1-score'],
                    'precision_non_tea': report['non_tea']['precision'],
                    'recall_non_tea': report['non_tea']['recall'],
                    'f1_non_tea': report['non_tea']['f1-score']
                })

            # Always save per‑image predictions
            pred_df = pd.DataFrame({
                'filename': [img['filename'] for img in val_images],
                'true_label': val_labels if have_labels else None,
                'predicted_label': y_pred,
                'probability_tea': y_proba if y_proba is not None else None
            })
            pred_df.to_csv(os.path.join(results_dir, 'validation', f'{model_name.lower()}_val_predictions.csv'),
                           index=False)

        except Exception as e:
            print(f"❌ Error validating {model_name}: {e}")
            traceback.print_exc()

    # Summary of all models (only if we had labels)
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(results_dir, 'validation', 'validation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print(summary_df.to_string(index=False))
        print(f"\n✅ Validation summary saved to: {summary_path}")
    else:
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE (no metrics computed, only predictions saved)")
        print("=" * 70)


# ---------------------------- MODIFIED MAIN BLOCK ----------------------------
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("TEA LEAF DETECTION SYSTEM")
    print("=" * 70)
    print("\nOptions:")
    print("1. Train models on tea leaf dataset")
    print("2. Test a single image")
    print("3. Validate on validation folder (requires trained models)")

    choice = input("\nEnter choice (1, 2, or 3, default=1): ").strip()

    if choice == "2":
        image_path = input("Enter image path to test: ").strip()
        if os.path.exists(image_path):
            main_with_prediction(image_path)
        else:
            print(f"Error: Image not found at {image_path}")
            print("Running training instead...")
            main_with_prediction()
    elif choice == "3":
        # Determine a sensible default validation folder
        if os.path.exists('data/validation'):
            default_val = 'data/validation'
        else:
            default_val = 'validation'
        val_folder = input(f"Enter validation folder path (default='{default_val}'): ").strip()
        if not val_folder:
            val_folder = default_val
        if not os.path.exists(val_folder):
            print(f"❌ Validation folder '{val_folder}' does not exist.")
        else:
            validate_models(val_folder, 'results')
    else:
        main_with_prediction()