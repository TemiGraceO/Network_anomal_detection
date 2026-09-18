from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('congestion_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inbound_rate = float(request.form['inbound_rate'])
    outbound_rate = float(request.form['outbound_rate'])
    inbound_util = float(request.form['inbound_util'])
    outbound_util = float(request.form['outbound_util'])

    data = np.array([[inbound_rate, outbound_rate, inbound_util, outbound_util]])
    prediction = model.predict(data)[0]

    result = "CONGESTED (Anomaly - Needs Rerouting)" if prediction == 1 else "NORMAL (Traffic OK)"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)