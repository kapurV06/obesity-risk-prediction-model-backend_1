from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model and tools
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
model = joblib.load(os.path.join(MODEL_DIR, 'final_obesity_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.pkl'))
feature_encoders = joblib.load(os.path.join(MODEL_DIR, 'feature_encoders.pkl'))
target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

# Lifestyle suggestions
def generate_suggestions(data):
    suggestions = []

    faf = float(data.get("FAF", 0))
    tue = float(data.get("TUE", 0))
    fcvc = int(float(data.get("FCVC", 0)))
    ch2o = float(data.get("CH2O", 0))
    calc = data.get("CALC", "").lower()
    mtrans = data.get("MTRANS", "").lower()

    if faf < 1:
        suggestions.append("Exercise ≥30 min/day (WHO)")
    elif faf < 2.5:
        suggestions.append("Increase weekly activity to 150–300 min (WHO)")

    if tue > 3:
        suggestions.append("Limit screen time <2 hrs/day (CDC)")

    if fcvc == 1:
        suggestions.append("Add vegetables to daily meals (Harvard)")
    elif fcvc == 2:
        suggestions.append("Increase veggie intake frequency (Harvard)")

    if ch2o < 1.5:
        suggestions.append("Drink 1.5–2L water/day (ICMR)")

    if "frequently" in calc or "always" in calc:
        suggestions.append("Limit alcohol intake (NHS UK)")

    if any(kw in mtrans for kw in ["motorbike", "automobile", "public"]):
        suggestions.append("Walk/cycle for short trips (WHO)")

    if not suggestions:
        suggestions.append("You're following healthy lifestyle habits. ✅")

    return suggestions

# Root endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Obesity Risk Prediction API is live.",
        "endpoints": {
            "/predict": "POST request with JSON body to get obesity risk class"
        }
    })

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        df_input = pd.DataFrame([input_data])

        # Cast numerics
        numeric_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        for col in numeric_columns:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(float)

        # Convert cm to meters if needed
        if df_input["Height"].max() > 10:
            df_input["Height"] = df_input["Height"] / 100.0

        weight = df_input["Weight"].values[0]
        height = df_input["Height"].values[0]
        bmi = round(weight / (height ** 2), 2)

        # Encode categorical features
        for col, encoder in feature_encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col] = encoder.transform(df_input[col])
                except Exception as e:
                    return jsonify({
                        "error": f"Invalid value for '{col}': {df_input[col].values[0]}",
                        "expected": list(encoder.classes_)
                    }), 400
            else:
                return jsonify({"error": f"Missing required field: {col}"}), 400

        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_columns]

        input_scaled = scaler.transform(df_input)
        pred_encoded = model.predict(input_scaled)[0]
        model_label = target_encoder.inverse_transform([pred_encoded])[0]

        # BMI-based correction
        if bmi < 18.5:
            corrected_label = "Insufficient_Weight"
        elif bmi < 25:
            corrected_label = "Normal_Weight"
        elif bmi < 30:
            corrected_label = "Overweight_Level_I"
        elif bmi < 35:
            corrected_label = "Obesity_Type_I"
        elif bmi < 40:
            corrected_label = "Obesity_Type_II"
        else:
            corrected_label = "Obesity_Type_III"

        category_map = {
            "Insufficient_Weight": "Underweight",
            "Normal_Weight": "Healthy",
            "Overweight_Level_I": "Borderline Obese",
            "Overweight_Level_II": "Borderline Obese",  # not used in BMI logic, but okay
            "Obesity_Type_I": "Obese (Class 1)",
            "Obesity_Type_II": "Obese (Class 2)",
            "Obesity_Type_III": "Obese (Morbid)"
        }

        # Safe retrieval
        readable_category = category_map.get(str(corrected_label).strip(), "Unknown")

        if readable_category == "Unknown":
            print(f"⚠️ DEBUG: corrected_label not in category_map: '{corrected_label}'")

        suggestions = generate_suggestions(input_data)

        print("✅ DEBUG:", {
            "BMI": bmi,
            "Model Label": model_label,
            "Corrected Label": corrected_label,
            "Readable Category": readable_category
        })
        print("\n=== DEBUG LOG START ===")
        print("📩 Raw input JSON:", input_data)
        print("📏 BMI value:", bmi)
        print("✅ Corrected Label:", corrected_label)
        print("📚 category_map keys:", list(category_map.keys()))
        print("🎯 Readable Category:", readable_category)
        print("=== DEBUG LOG END ===\n")

        return jsonify({
            "model_prediction": model_label,
            "corrected_prediction": corrected_label,
            "category": readable_category,
            "bmi": bmi,
            "suggestions": suggestions,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
