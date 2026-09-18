import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'congestion_model.pkl')
TEMPLATE_PATH = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_PATH)
model = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # These 4 names MUST match your input fields in HTML
        data = {
            'packets_per_sec': float(request.form.get('packets_per_sec', 0)),
            'avg_packet_size': float(request.form.get('avg_packet_size', 0)),
            'latency_ms': float(request.form.get('latency_ms', 0)),
            'bandwidth_util': float(request.form.get('bandwidth_util', 0))
        }

        df = pd.DataFrame([data])
        result = model.predict(df)[0]

        if result == 1:
            prediction = "CONGESTED - Anomaly Detected"
        else:
            prediction = "NORMAL - No Congestion"

    except Exception as e:
        prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)