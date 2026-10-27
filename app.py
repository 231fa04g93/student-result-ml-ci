```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "student-result-prediction",
        "status": "ok"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Provide valid JSON input"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    try:
        values = [float(data[feature]) for feature in FEATURES]
    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    if not np.all(np.isfinite(values)):
        return jsonify({
            "error": "Feature values must be finite numbers"
        }), 400

    # Create input using the same feature names and order
    input_data = pd.DataFrame([values], columns=FEATURES)

    prediction = int(model.predict(input_data)[0])
    result = "PASS" if prediction == 1 else "FAIL"

    return jsonify({
        "prediction": result,
        "prediction_code": prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```
