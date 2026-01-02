from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        fields = [
            "gender", "age", "smoking", "yellow_fingers", "anxiety",
            "peer_pressure", "chronic_disease", "fatigue", "allergy",
            "wheezing", "alcohol", "coughing", "shortness_breath",
            "swallowing_difficulty", "chest_pain"
        ]

        # Validate empty inputs
        for field in fields:
            if request.form[field].strip() == "":
                return render_template(
                    "index.html",
                    prediction_text="⚠️ Please fill in all fields before predicting"
                )

        # Convert inputs to float
        features = [float(request.form[field]) for field in fields]

        # Scale features
        scaled_features = scaler.transform([features])

        # Get probability (for class = 1 → Cancer)
        probability = model.predict_proba(scaled_features)[0][1]

        # Probability-based decision
        if probability >= 0.7:
            result = f"🔴 High Risk: Lung Cancer Detected (Probability: {probability:.2f})"
        elif probability >= 0.4:
            result = f"🟠 Moderate Risk: Medical Check Recommended (Probability: {probability:.2f})"
        else:
            result = f"🟢 Low Risk: No Lung Cancer (Probability: {probability:.2f})"

        return render_template("index.html", prediction_text=result)

    except Exception:
        return render_template(
            "index.html",
            prediction_text="⚠️ Invalid input. Please enter numeric values only."
        )


if __name__ == "__main__":
    app.run(debug=True)
