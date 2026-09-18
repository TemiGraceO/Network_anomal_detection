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
        data = {
            'Inbound Rate(bit/s)': float(request.form.get('inbound_rate', 0)),
            'Outbound Rate(bit/s)': float(request.form.get('outbound_rate', 0)),
            'Inbound Bandwidth Utilization(%)': float(request.form.get('inbound_bw', 0)),
            'Outbound Bandwidth Utilization(%)': float(request.form.get('outbound_bw', 0))
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