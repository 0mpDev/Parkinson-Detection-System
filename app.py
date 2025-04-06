from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
import joblib
import os

app = Flask(__name__)

# Load the model and transformers
def load_model():
    try:
        if os.path.exists('model.pkl'):
            return joblib.load('model.pkl')
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_scaler():
    try:
        if os.path.exists('scaler.pkl'):
            return joblib.load('scaler.pkl')
        return None
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        return None

def load_selector():
    try:
        if os.path.exists('selector.pkl'):
            return joblib.load('selector.pkl')
        return None
    except Exception as e:
        print(f"Error loading selector: {str(e)}")
        return None

def load_pca():
    try:
        if os.path.exists('pca.pkl'):
            return joblib.load('pca.pkl')
        return None
    except Exception as e:
        print(f"Error loading PCA: {str(e)}")
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
        
        # Load model and transformers
        model = load_model()
        scaler = load_scaler()
        selector = load_selector()
        pca = load_pca()
        
        if model is None or scaler is None or selector is None or pca is None:
            return jsonify({'error': 'Model or transformers not found. Please train the model first.'})
        
        # Transform the features in the correct order
        features_selected = selector.transform(features)  # First apply feature selection
        features_scaled = scaler.transform(features_selected)  # Then scale the selected features
        features_pca = pca.transform(features_scaled)  # Finally apply PCA
        
        # Make prediction
        prediction = model.predict(features_pca)[0]
        probability = model.predict_proba(features_pca)[0][1]
        
        result = {
            'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            'probability': f'{probability:.2%}',
            'status': 'success'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'An error occurred during prediction: {str(e)}',
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001) 