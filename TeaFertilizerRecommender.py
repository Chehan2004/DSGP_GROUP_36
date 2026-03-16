import os
import sys

import joblib
import pandas as pd


# Fix sklearn pickle compatibility
class _RemainderColsList(list):
    pass

sys.modules["__main__"]._RemainderColsList = _RemainderColsList


class TeaFertilizerRecommender:
    """
    Class to load tea fertilizer ML models and predict fertilizer + dosage
    """

    def __init__(self, models_dir="models/fertilizer_models"):
        """
        Load all models and label encoders from the specified directory.
        Args:
            models_dir (str): Directory where the models and encoders are stored.
        Raises:
            FileNotFoundError: If any of the model or encoder files are missing.
        """
        # Paths
        self.models_paths = {
            "decision_tree": os.path.join(models_dir, "tea_dt_pipeline.pkl"),
            "random_forest": os.path.join(models_dir, "tea_rf_pipeline.pkl"),
            "gradient_boosting": os.path.join(models_dir, "tea_gb_pipeline.pkl")
        }
        self.le_fert_path = os.path.join(models_dir, "le_fert.pkl")
        self.le_dose_path = os.path.join(models_dir, "le_dose.pkl")

        # Load models
        self.models = {name: joblib.load(path) for name, path in self.models_paths.items()}

        # Load encoders
        self.le_fert = joblib.load(self.le_fert_path)
        self.le_dose = joblib.load(self.le_dose_path)

        print("Models loaded:", list(self.models.keys()))

        if not self.models:
            raise FileNotFoundError("No models found in the specified directory.")
        if not self.le_fert or not self.le_dose:
            raise FileNotFoundError("Label encoders not found in the specified directory.")


    # Helper method to convert encoded predictions back to original labels
    @staticmethod
    def format_dosage(dosage: str) -> str:

        # Map encoded dosage to human-readable format
        dosage_map = {
            "low": "Low (≈80 kg/ha per application)",
            "medium": "Medium (≈120 kg/ha per application)",
            "high": "High (≈160 kg/ha per application)"
        }
        # Return the human-readable dosage or the original if not found in the map
        return dosage_map.get(dosage.lower(), dosage)

    def predict(self, soil_type, soil_pH, rainfall_mm_week, humidity_percent,
                temperature_c, disease, disease_severity, model):

        """

        Predict the recommended fertilizer and dosage for a tea plant
            based on soil conditions, weather factors, and detected disease.

            The function builds an input record from the provided parameters,
            feeds it into a selected trained machine learning pipeline, and
            decodes the predicted fertilizer type and dosage from their
            encoded form back to human-readable values.

            Parameters
            ----------
            soil_type : str
                Type of soil (e.g., loamy, sandy, clay).
            soil_pH : float
                Average soil pH level.
            rainfall_mm_week : float
                Weekly rainfall in millimeters.
            humidity_percent : float
                Average humidity percentage.
            temperature_c : float
                Average temperature in Celsius.
            disease : str
                Detected tea leaf disease name.
            disease_severity : str
                Severity level of the disease (e.g., low, medium, high).
            model : str
                Name/key of the trained prediction pipeline stored in
                self.models.

            Returns
            -------
            Tuple[str, str]
                fertilizer : str
                    Recommended fertilizer type.
                dosage : str
                    Recommended dosage in formatted human-readable form.

        """

        # Validate model choice
        if model not in self.models:
            raise ValueError(f"Model '{model}' not available")

        # Create dataframe
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


import os
import sys

import joblib
import pandas as pd


# Fix sklearn pickle compatibility
class _RemainderColsList(list):
    pass

sys.modules["__main__"]._RemainderColsList = _RemainderColsList


class TeaFertilizerRecommender:
    """
    Class to load tea fertilizer ML models and predict fertilizer + dosage
    """

    def __init__(self, models_dir="models/fertilizer_models"):
        """
        Load all models and label encoders from the specified directory.
        Args:
            models_dir (str): Directory where the models and encoders are stored.
        Raises:
            FileNotFoundError: If any of the model or encoder files are missing.
        """
        # Paths
        self.models_paths = {
            "decision_tree": os.path.join(models_dir, "tea_dt_pipeline.pkl"),
            "random_forest": os.path.join(models_dir, "tea_rf_pipeline.pkl"),
            "gradient_boosting": os.path.join(models_dir, "tea_gb_pipeline.pkl")
        }
        self.le_fert_path = os.path.join(models_dir, "le_fert.pkl")
        self.le_dose_path = os.path.join(models_dir, "le_dose.pkl")

        # Load models
        self.models = {name: joblib.load(path) for name, path in self.models_paths.items()}

        # Load encoders
        self.le_fert = joblib.load(self.le_fert_path)
        self.le_dose = joblib.load(self.le_dose_path)

        print("Models loaded:", list(self.models.keys()))

        if not self.models:
            raise FileNotFoundError("No models found in the specified directory.")
        if not self.le_fert or not self.le_dose:
            raise FileNotFoundError("Label encoders not found in the specified directory.")


    # Helper method to convert encoded predictions back to original labels
    @staticmethod
    def format_dosage(dosage: str) -> str:

        # Map encoded dosage to human-readable format
        dosage_map = {
            "low": "Low (≈80 kg/ha per application)",
            "medium": "Medium (≈120 kg/ha per application)",
            "high": "High (≈160 kg/ha per application)"
        }
        # Return the human-readable dosage or the original if not found in the map
        return dosage_map.get(dosage.lower(), dosage)

    def predict(self, soil_type, soil_pH, rainfall_mm_week, humidity_percent,
                temperature_c, disease, disease_severity, model):

        """

        Predict the recommended fertilizer and dosage for a tea plant
            based on soil conditions, weather factors, and detected disease.

            The function builds an input record from the provided parameters,
            feeds it into a selected trained machine learning pipeline, and
            decodes the predicted fertilizer type and dosage from their
            encoded form back to human-readable values.

            Parameters
            ----------
            soil_type : str
                Type of soil (e.g., loamy, sandy, clay).
            soil_pH : float
                Average soil pH level.
            rainfall_mm_week : float
                Weekly rainfall in millimeters.
            humidity_percent : float
                Average humidity percentage.
            temperature_c : float
                Average temperature in Celsius.
            disease : str
                Detected tea leaf disease name.
            disease_severity : str
                Severity level of the disease (e.g., low, medium, high).
            model : str
                Name/key of the trained prediction pipeline stored in
                self.models.

            Returns
            -------
            Tuple[str, str]
                fertilizer : str
                    Recommended fertilizer type.
                dosage : str
                    Recommended dosage in formatted human-readable form.

        """

        # Validate model choice
        if model not in self.models:
            raise ValueError(f"Model '{model}' not available")

        # Create dataframe
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
    # Initialize the recommender
    # Ensure your models are in 'models/fertilizer_models' as per the class default
    try:
        recommender = TeaFertilizerRecommender()

        # Sample input data for testing
        test_data = {
            "soil_type": "Loamy",
            "soil_pH": 6.2,
            "rainfall_mm_week": 45.0,
            "humidity_percent": 80.0,
            "temperature_c": 24.5,
            "disease": "Blister Blight",
            "disease_severity": "Medium",
            "model": "random_forest"  # Options: 'decision_tree', 'random_forest', 'gradient_boosting'
        }

        print("-" * 40)
        print("Tea Fertilizer Recommendation Test")
        print("-" * 40)
        print(f"Input: {test_data['disease']} ({test_data['disease_severity']} severity)")

        # Get recommendation
        fertilizer, dosage = recommender.predict(**test_data)

        print(f"Recommended Fertilizer : {fertilizer}")
        print(f"Recommended Dosage     : {dosage}")
        print("-" * 40)

    except FileNotFoundError as e:
        print(f"Error: Could not find model files. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
