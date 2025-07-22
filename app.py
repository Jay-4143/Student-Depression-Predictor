from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Paths to required files
MODEL_PATH = "depression_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"

# Load model components safely
try:
    if not all(os.path.exists(path) for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        raise FileNotFoundError("One or more required model files are missing.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    raise

# Cluster interpretation
CLUSTER_LABELS = {
    0: "Depressed",
    1: "Not Depressed"
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running. Use POST /predict with JSON input."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        df = pd.DataFrame([data])
        df = df.reindex(columns=features, fill_value=0)
        scaled_input = scaler.transform(df)
        cluster = model.predict(scaled_input)[0]

        prediction = CLUSTER_LABELS.get(cluster, "Unknown")
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
