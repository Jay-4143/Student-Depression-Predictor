from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load everything
model = joblib.load("depression_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_columns.pkl")

# Determine correct mapping for cluster -> label
# NOTE: Update based on analysis if needed
CLUSTER_LABELS = {
    0: "Depressed",
    1: "Not Depressed"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)
    scaled_input = scaler.transform(df)
    cluster = model.predict(scaled_input)[0]

    prediction = CLUSTER_LABELS.get(cluster, "Unknown")
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
