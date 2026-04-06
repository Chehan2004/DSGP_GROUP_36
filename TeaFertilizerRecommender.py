import sklearn.compose._column_transformer as _ct
import __main__



# This class is needed to unpickle pipelines that were saved with older versions of sklearn (pre-1.2) where the _RemainderColsList class was defined in a different location. By defining it here, we ensure that we can load those older pipelines without errors. The class itself is a simple subclass of list that includes an additional attribute for future compatibility, but it doesn't affect the core functionality of the pipelines.
class _RemainderColsList(list):
    def __init__(self, lst, future_dtype=None):
        super().__init__(lst)
        self.future_dtype = future_dtype


# Assign the _RemainderColsList class to both the sklearn.compose._column_transformer module and the __main__ module so that it can be found when unpickling older pipelines.
_ct._RemainderColsList = _RemainderColsList
__main__._RemainderColsList = _RemainderColsList


import joblib
import pandas as pd


class TeaFertilizerRecommender:

    DOSAGE_DESCRIPTIONS = {
        "low":    "Low (≈80–100 kg/ha per application)",
        "normal": "Normal (≈110–130 kg/ha per application)",
        "split":  "Split application (apply ≈110–130 kg/ha in multiple smaller doses)",
        "delay":  "Delay application (postpone fertilizer due to unfavorable conditions)",
    }

    DISEASE_RECOMMENDATIONS = {
        "algal leaf":         {"type": "fungal",        "treatment": "Copper hydroxide / Copper oxychloride"},
        "anthracnose":        {"type": "fungal",        "treatment": "Copper oxychloride"},
        "bird eye spot":      {"type": "fungal",        "treatment": "Copper oxychloride"},
        "brown blight":       {"type": "fungal",        "treatment": "Copper oxychloride"},
        "gray blight":        {"type": "fungal",        "treatment": "Copper oxychloride"},
        "green mirid bug":    {"type": "pest",          "treatment": "Insect pest – use IPM and consult local guidelines"},
        "heliopeltis":        {"type": "pest",          "treatment": "Insect pest – use IPM and consult local guidelines"},
        "red rust":           {"type": "fungal",        "treatment": "Copper hydroxide / Cuprous oxide"},
        "red spider":         {"type": "pest",          "treatment": "Sulphur"},
        "sunlight scorching": {"type": "physiological", "treatment": "No pesticide – improve shade and irrigation"},
        "tea leaf blight":    {"type": "fungal",        "treatment": "Copper hydroxide"},
        "tea red leaf spot":  {"type": "fungal",        "treatment": "Copper-based fungicide (confirm in field)"},
        "tea red scab":       {"type": "fungal",        "treatment": "Copper-based fungicide (confirm in field)"},
        "thrips":             {"type": "pest",          "treatment": "Use IPM – confirm before applying insecticide"},
        "white spot":         {"type": "fungal",        "treatment": "Copper oxychloride"},
    }

    MODEL_FILES = {
        "dt": "models/fertilizer_models/tea_dt_pipeline.pkl",
        "rf": "models/fertilizer_models/tea_rf_pipeline.pkl",
        "gb": "models/fertilizer_models/tea_gb_pipeline_new.pkl",
    }

    def __init__(self):
        self.models = {}
        self.le_fert = None
        self.le_dose = None
        self._load_resources()

    def _load_resources(self):
        for key, filename in self.MODEL_FILES.items():
            self.models[key] = joblib.load(filename)
        self.le_fert = joblib.load("models/fertilizer_models/le_fert.pkl")
        self.le_dose = joblib.load("models/fertilizer_models/le_dose.pkl")

    def format_dosage(self, dosage: str) -> str:
        key = str(dosage).lower()
        return self.DOSAGE_DESCRIPTIONS.get(key, dosage)

    def get_disease_info(self, disease: str) -> dict:
        return self.DISEASE_RECOMMENDATIONS.get(disease.lower(), {})

    def predict(self, model_key: str, soil_type: str, soil_pH: float,
                rainfall_mm_week: float, humidity_percent: float,
                temperature_c: float, disease: str, disease_severity: str) -> dict:
        """
        Predict fertilizer recommendation and dosage based on input features using the specified model.
        Parameters:
            model_key (str): Key to select the model ('dt', 'rf', 'gb').
            soil_type (str): Type of soil (e.g., 'Loamy', 'Sandy', 'Clay').
            soil_pH (float): pH level of the soil.
            rainfall_mm_week (float): Average rainfall in mm per week.      
            humidity_percent (float): Average humidity in percentage.
            temperature_c (float): Average temperature in degrees Celsius.
            disease (str): Name of the disease affecting the tea plants.
            disease_severity (str): Severity of the disease (e.g., 'Low', 'Medium', 'High').
        Returns:
            dict: A dictionary containing the predicted fertilizer, dosage recommendation, confidence score (if available), and disease information (type and treatment)
        """

        if model_key not in self.models:
            raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(self.models.keys())}")

        input_df = pd.DataFrame([{
            "soil_type":        soil_type,
            "soil_pH":          soil_pH,
            "rainfall_mm_week": rainfall_mm_week,
            "humidity_percent": humidity_percent,
            "temperature_c":    temperature_c,
            "disease":          disease,
            "disease_severity": disease_severity,
        }])

        pipeline   = self.models[model_key]
        prediction = pipeline.predict(input_df)[0]

        fertilizer = self.le_fert.inverse_transform([prediction[0]])[0]
        dosage_raw = self.le_dose.inverse_transform([prediction[1]])[0]
        dosage     = self.format_dosage(dosage_raw)

        confidence = None
        try:
            proba     = pipeline.predict_proba(input_df)
            fert_conf = float(proba[0].max(axis=1)[0])
            dose_conf = float(proba[1].max(axis=1)[0])
            confidence = round((fert_conf + dose_conf) / 2, 3)
        except Exception:
            pass

        disease_info = self.get_disease_info(disease)

        return {
            "fertilizer":   fertilizer,
            "dosage":       dosage,
            "confidence":   confidence,
            "disease_type": disease_info.get("type"),
            "treatment":    disease_info.get("treatment"),
        }


# Test cases to validate the functionality of the TeaAdvisor class. Each test case includes a set of input features and prints the predicted fertilizer, dosage recommendation, confidence score, and disease information.
if __name__ == "__main__":

    TEST_CASES = [
        {
            "model_key":        "rf",
            "soil_type":        "Loamy",
            "soil_pH":          5.5,
            "rainfall_mm_week": 100.0,
            "humidity_percent": 80.0,
            "temperature_c":    25.0,
            "disease":          "Brown Blight",
            "disease_severity": "Medium - Seen in less than 10 plants",
        },
        {
            "model_key":        "gb",
            "soil_type":        "Sandy",
            "soil_pH":          6.0,
            "rainfall_mm_week": 60.0,
            "humidity_percent": 70.0,
            "temperature_c":    28.0,
            "disease":          "Red Spider",
            "disease_severity": "High - Seen in more than 10 plants",
        },
        {
            "model_key":        "dt",
            "soil_type":        "Clay",
            "soil_pH":          4.8,
            "rainfall_mm_week": 140.0,
            "humidity_percent": 90.0,
            "temperature_c":    22.0,
            "disease":          "Algal Leaf",
            "disease_severity": "Low - Seen in less than 3 plants",
        },
    ]

    advisor = TeaFertilizerRecommender()

    for i, tc in enumerate(TEST_CASES, start=1):
        print(f"\n{'='*55}")
        print(f" Test Case {i}  |  Model: {tc['model_key'].upper()}")
        print(f"{'='*55}")
        print(f"  Soil Type   : {tc['soil_type']}")
        print(f"  Soil pH     : {tc['soil_pH']}")
        print(f"  Rainfall    : {tc['rainfall_mm_week']} mm/week")
        print(f"  Humidity    : {tc['humidity_percent']}%")
        print(f"  Temperature : {tc['temperature_c']} °C")
        print(f"  Disease     : {tc['disease']}")
        print(f"  Severity    : {tc['disease_severity']}")
        print(f"{'-'*55}")

        result = advisor.predict(**tc)

        print(f"  Fertilizer  : {result['fertilizer']}")
        print(f"  Dosage      : {result['dosage']}")
        if result["confidence"] is not None:
            print(f"  Confidence  : {result['confidence']}")
        if result["disease_type"]:
            print(f"  Disease Type: {result['disease_type']}")
        if result["treatment"]:
            print(f"  Treatment   : {result['treatment']}")

    print(f"\n{'='*55}\n")