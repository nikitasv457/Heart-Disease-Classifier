from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get user input from form
            user_input = [float(request.form[f"feature_{i}"]) for i in range(1, 14)]  # Needs to be adjusted for dataset ----------------
            user_input = np.array(user_input).reshape(1, -1)
            
            # Standardize input
            user_input = scaler.transform(user_input)

            # Predict using model
            prediction = model.predict(user_input)[0]
            prediction = "Positive (Heart Disease)" if prediction == 1 else "Negative (No Heart Disease)"
        
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
