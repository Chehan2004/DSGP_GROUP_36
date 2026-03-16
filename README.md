### Project Overview

TeaVision is an automated system for assessing tea leaves. It uses machine learning to detect leaves, identify diseases, grade quality, and recommend fertilizers.

### Key Components

- **App Entry**: `app.py` serves as the main interface for the user.
- **Detection**: `TeaDetector.py` localizes tea leaves within images.
- **Classification**: `TeaDiseaseClassifier.py` identifies specific diseases on the leaves.
- **Grading**: `QualityGrader.py` assesses the quality of the tea leaves.
- **Recommendation**: `TeaFertilizerRecommender.py` provides fertilizer suggestions based on leaf data.

### Deployment Requirements

- **Python Libraries**: Managed via `requirements.txt`.
- **System Dependencies**: Linux requirements are listed in `packages.txt`.
- **Styling**: Custom UI elements are defined in `style.css`.
- **Exclusions**: Large model weights and data files are excluded from the repository via `.gitignore`.
