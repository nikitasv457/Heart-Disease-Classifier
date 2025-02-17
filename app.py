from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence_no_disease = 0
    confidence_heart_disease = 0

    if request.method == "POST":
        try:
            # Get user input from form
            user_input = [float(request.form[f"feature_{i}"]) for i in range(1, 14)]
            user_input = np.array(user_input).reshape(1, -1)
            
            # Standardize input
            user_input = scaler.transform(user_input)

            # Predict using model
            proba = model.predict_proba(user_input)[0]  # Get confidence probabilities
            prediction = "Positive (Heart Disease)" if proba[1] > 0.5 else "Negative (No Heart Disease)"

            #   Confidence levels for graphs
            confidence_no_disease = round(proba[0] * 100, 2)
            confidence_heart_disease = round(proba[1] * 100, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction,
                           confidence_no_disease=confidence_no_disease,
                           confidence_heart_disease=confidence_heart_disease)

# API for AJAX Requests 
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        user_input = np.array([data["features"]]).reshape(1, -1)
        user_input = scaler.transform(user_input)

        # Predict
        proba = model.predict_proba(user_input)[0]
        prediction = "Positive (Heart Disease)" if proba[1] > 0.5 else "Negative (No Heart Disease)"
        confidence = {"no_disease": round(proba[0] * 100, 2), "heart_disease": round(proba[1] * 100, 2)}

        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
