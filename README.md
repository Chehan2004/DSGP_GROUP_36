### Project Overview

TeaVision is an automated system for assessing tea leaves. It uses machine learning to detect leaves, identify diseases, grade quality, and recommend fertilizers.

### Key Components

- app.py: Main user interface.
- TeaLeafPredictor.py: Primary inference engine for "Leaf or Not" binary validation and leaf assessment.
- TeaDetector.py: Localization and region extraction.
- TeaDiseaseClassifier.py: Multi-class disease identification.
- QualityGrader.py: Neural network grading (T0-T4).
- TeaFertilizerRecommender.py: Fertilizer dosage suggestions.

### Deployment Requirements

- **Python Libraries**: Managed via `requirements.txt`.
- **System Dependencies**: Linux requirements are listed in `packages.txt`.
- **Styling**: Custom UI elements are defined in `style.css`.
- **Exclusions**: Large model weights and data files are excluded from the repository via `.gitignore`.
- Version: TeaVision_v01_20260316_Base
