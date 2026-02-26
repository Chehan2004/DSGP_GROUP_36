from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
pipeline = joblib.load("tea_dt_pipeline.pkl")
le_fert = joblib.load("le_fert.pkl")
le_dose = joblib.load("le_dose.pkl")

@app.route("/")
def home():
    return {"message": "Tea Fertilizer Recommendation API is running"}

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([data])

    prediction = pipeline.predict(input_df)[0]

    fertilizer = le_fert.inverse_transform([prediction[0]])[0]
    dosage = le_dose.inverse_transform([prediction[1]])[0]

    print("RAW prediction:", prediction)
    print("Decoded fertilizer:", fertilizer)
    print("Decoded dosage:", dosage)

    # Convert to strings
    fertilizer = str(fertilizer)
    dosage = str(dosage)
    if dosage.lower() == "low":
        dosage = "Low (≈80 kg/ha per application)"
    elif dosage.lower() == "medium":
        dosage = "Medium (≈120 kg/ha per application)"
    elif dosage.lower() == "high":
        dosage = "High (≈160 kg/ha per application)"

    return jsonify({
        "recommended_fertilizer": fertilizer,
        "recommended_fertilizer_dosage": dosage
    })

if __name__ == "__main__":
    app.run(debug=True)