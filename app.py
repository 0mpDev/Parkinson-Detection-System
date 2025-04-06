from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)

# Load the model and scaler
def load_model():
    if os.path.exists('model.pkl'):
        return joblib.load('model.pkl')
    return None

def load_scaler():
    if os.path.exists('scaler.pkl'):
        return joblib.load('scaler.pkl')
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [
            float(request.form['mdvp_fo']),
            float(request.form['mdvp_fhi']),
            float(request.form['mdvp_flo']),
            float(request.form['mdvp_jitter']),
            float(request.form['mdvp_jitter_abs']),
            float(request.form['mdvp_rap']),
            float(request.form['mdvp_ppq']),
            float(request.form['jitter_ddp']),
            float(request.form['mdvp_shimmer']),
            float(request.form['mdvp_shimmer_db']),
            float(request.form['shimmer_apq3']),
            float(request.form['shimmer_apq5']),
            float(request.form['mdvp_apq']),
            float(request.form['shimmer_dda']),
            float(request.form['nhr']),
            float(request.form['hnr']),
            float(request.form['rpde']),
            float(request.form['dfa']),
            float(request.form['spread1']),
            float(request.form['spread2']),
            float(request.form['d2']),
            float(request.form['ppe']),
        ]
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Load model and scaler
        model = load_model()
        scaler = load_scaler()
        
        if model is None or scaler is None:
            return jsonify({'error': 'Model not found. Please train the model first.'})
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        result = {
            'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            'probability': f'{probability:.2%}'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001) 