import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage import feature, filters, measure
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


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
            for filename in os.listdir(image_folder):
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
                shape_features, shape_names = self.extract_shape_size_features(img_data)  # NEW LINE

                # Combine all features
                features.extend(color_features)
                features.extend(texture_features)
                features.extend(shape_features)  # NEW LINE

                current_feature_names.extend(color_names)
                current_feature_names.extend(texture_names)
                current_feature_names.extend(shape_names)  # NEW LINE

                # Store feature names only once
                if not self.feature_names and current_feature_names:
                    self.feature_names = current_feature_names

                all_features.append(features)
                all_labels.append(img_data['label'])

            except Exception as e:
                print(f"Error processing image {img_data.get('filename', 'unknown')}: {e}")
                continue

        return np.array(all_features), np.array(all_labels)


def main():
    """
    Main function to run tea leaf feature engineering
    """
    # Create directories
    os.makedirs('results', exist_ok=True)

    # Initialize feature extractor
    extractor = TeaLeafFeatureExtractor()

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
    feature_df = pd.DataFrame(features, columns=extractor.feature_names)
    feature_df['label'] = labels
    feature_df['is_tea_leaf'] = ['Yes' if l == 1 else 'No' for l in labels]

    # Calculate feature importance
    print("Calculating feature importance...")
    X = feature_df.drop(['label', 'is_tea_leaf'], axis=1)
    y = feature_df['label']

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_scores = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': extractor.feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)

    # SAVE CSV FILES - THIS IS THE KEY PART!
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
    plt.show()
    print("✓ Saved: results/feature_importance.png")

    # Print summary
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"📊 Total images processed: {len(features)}")
    print(f"   - Tea leaves: {len(tea_images)}")
    print(f"   - Non-tea objects: {len(non_tea_images)}")
    print(f"🔧 Total features extracted: {features.shape[1]}")

    # Count feature types
    color_features = len([f for f in extractor.feature_names if
                          any(f.startswith(prefix) for prefix in ['rgb_', 'hsv_', 'lab_', 'green_'])])
    texture_features = len(
        [f for f in extractor.feature_names if any(f.startswith(prefix) for prefix in ['glcm_', 'lbp_', 'gray_'])])
    shape_features = len([f for f in extractor.feature_names if any(f.startswith(prefix) for prefix in
                                                                    ['contour_', 'bounding_', 'aspect_', 'circularity',
                                                                     'solidity', 'equivalent_', 'major_', 'minor_',
                                                                     'eccentricity', 'rectangularity', 'compactness',
                                                                     'hu_moment_'])])

    print(f"🎨 Color features: {color_features}")
    print(f"🔍 Texture features: {texture_features}")
    print(f"📐 Shape features: {shape_features}")

    print(f"\n📁 Files created in 'results' folder:")
    print(f"   - tea_leaf_features.csv (All features for {len(features)} images)")
    print(f"   - feature_importance_scores.csv (Importance for all {len(importance_df)} features)")
    print(f"   - top_20_features.csv (Only the 20 most important features)")
    print(f"   - feature_importance.png (Visualization of top 15 features)")

    print(f"\n🏆 Top 5 most important features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"   {i + 1}. {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()