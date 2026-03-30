import os
import sys

import joblib
import pandas as pd
import sklearn.compose._column_transformer as _ct


# Fix sklearn pickle compatibility
class _RemainderColsList(list):
    pass

# Patch in all locations sklearn might look
_ct._RemainderColsList = _RemainderColsList
sys.modules["__main__"]._RemainderColsList = _RemainderColsList
sys.modules[__name__]._RemainderColsList = _RemainderColsList


class TeaFertilizerRecommender:

    def __init__(self, models_dir="models/fertilizer_models"):
        self.models_paths = {
            "decision_tree": os.path.join(models_dir, "tea_dt_pipeline.pkl"),
            "random_forest": os.path.join(models_dir, "tea_rf_pipeline.pkl"),
            "gradient_boosting": os.path.join(models_dir, "tea_gb_pipeline_new.pkl")
        }
        self.le_fert_path = os.path.join(models_dir, "le_fert.pkl")
        self.le_dose_path = os.path.join(models_dir, "le_dose.pkl")

        self.models = {name: joblib.load(path) for name, path in self.models_paths.items()}

        self.le_fert = joblib.load(self.le_fert_path)
        self.le_dose = joblib.load(self.le_dose_path)

        print("Models loaded:", list(self.models.keys()))

        if not self.models:
            raise FileNotFoundError("No models found in the specified directory.")
        if not self.le_fert or not self.le_dose:
            raise FileNotFoundError("Label encoders not found in the specified directory.")

    @staticmethod
    def format_dosage(dosage: str) -> str:
        dosage_map = {
            "low": "Low (≈80–100 kg/ha per application)",
            "normal": "Normal (≈110–130 kg/ha per application)",
            "split": "Split application (apply ≈110–130 kg/ha in multiple smaller doses)",
            "delay": "Delay application (postpone fertilizer due to unfavorable conditions)"
        }
        return dosage_map.get(dosage.lower(), dosage)

    def predict(self, soil_type, soil_pH, rainfall_mm_week, humidity_percent,
                temperature_c, disease, disease_severity, model):

        if model not in self.models:
            raise ValueError(f"Model '{model}' not available")

        input_df = pd.DataFrame([{
            "soil_type": soil_type,
            "soil_pH": soil_pH,
            "rainfall_mm_week": rainfall_mm_week,
            "humidity_percent": humidity_percent,
            "temperature_c": temperature_c,
            "disease": disease,
            "disease_severity": disease_severity
        }])

        pipeline = self.models[model]
        pred = pipeline.predict(input_df)[0]

        fertilizer = self.le_fert.inverse_transform([pred[0]])[0]
        dosage_raw = self.le_dose.inverse_transform([pred[1]])[0]
        dosage = self.format_dosage(dosage_raw)

        return fertilizer, dosage


if __name__ == "__main__":
    try:
        recommender = TeaFertilizerRecommender()

        test_data = {
            "soil_type": "Loamy",
            "soil_pH": 6.2,
            "rainfall_mm_week": 45.0,
            "humidity_percent": 80.0,
            "temperature_c": 24.5,
            "disease": "Brown Blight",
            "disease_severity": "Medium - Seen in less than 10 plants",
            "model": "random_forest"
        }

        print("-" * 40)
        print("Tea Fertilizer Recommendation Test")
        print("-" * 40)
        print(f"Input: {test_data['disease']} ({test_data['disease_severity']} severity)")

        fertilizer, dosage = recommender.predict(**test_data)

        print(f"Recommended Fertilizer : {fertilizer}")
        print(f"Recommended Dosage     : {dosage}")
        print("-" * 40)

    except FileNotFoundError as e:
        print(f"Error: Could not find model files. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")