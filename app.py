# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectKBest
# import joblib
# import os

# app = Flask(__name__)

# # Load the model and transformers
# def load_model():
#     try:
#         if os.path.exists('model.pkl'):
#             return joblib.load('model.pkl')
#         return None
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         return None

# def load_scaler():
#     try:
#         if os.path.exists('scaler.pkl'):
#             return joblib.load('scaler.pkl')
#         return None
#     except Exception as e:
#         print(f"Error loading scaler: {str(e)}")
#         return None

# def load_selector():
#     try:
#         if os.path.exists('selector.pkl'):
#             return joblib.load('selector.pkl')
#         return None
#     except Exception as e:
#         print(f"Error loading selector: {str(e)}")
#         return None

# def load_pca():
#     try:
#         if os.path.exists('pca.pkl'):
#             return joblib.load('pca.pkl')
#         return None
#     except Exception as e:
#         print(f"Error loading PCA: {str(e)}")
#         return None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get values from the form
#         features = [
#             float(request.form['mdvp_fo']),
#             float(request.form['mdvp_fhi']),
#             float(request.form['mdvp_flo']),
#             float(request.form['mdvp_jitter']),
#             float(request.form['mdvp_jitter_abs']),
#             float(request.form['mdvp_rap']),
#             float(request.form['mdvp_ppq']),
#             float(request.form['jitter_ddp']),
#             float(request.form['mdvp_shimmer']),
#             float(request.form['mdvp_shimmer_db']),
#             float(request.form['shimmer_apq3']),
#             float(request.form['shimmer_apq5']),
#             float(request.form['mdvp_apq']),
#             float(request.form['shimmer_dda']),
#             float(request.form['nhr']),
#             float(request.form['hnr']),
#             float(request.form['rpde']),
#             float(request.form['dfa']),
#             float(request.form['spread1']),
#             float(request.form['spread2']),
#             float(request.form['d2']),
#             float(request.form['ppe']),
#         ]
        
#         # Convert to numpy array and reshape
#         features = np.array(features).reshape(1, -1)
        
#         # Load model and transformers
#         model = load_model()
#         scaler = load_scaler()
#         selector = load_selector()
#         pca = load_pca()
        
#         if model is None or scaler is None or selector is None or pca is None:
#             return jsonify({'error': 'Model or transformers not found. Please train the model first.'})
        
#         # Transform the features in the correct order
#         features_selected = selector.transform(features)  # First apply feature selection
#         features_scaled = scaler.transform(features_selected)  # Then scale the selected features
#         features_pca = pca.transform(features_scaled)  # Finally apply PCA
        
#         # Make prediction
#         prediction = model.predict(features_pca)[0]
#         probability = model.predict_proba(features_pca)[0][1]
        
#         result = {
#             'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
#             'probability': f'{probability:.2%}',
#             'status': 'success'
#         }
        
#         return jsonify(result)
    
#     except Exception as e:
#         print(f"Prediction error: {str(e)}")
#         return jsonify({
#             'error': f'An error occurred during prediction: {str(e)}',
#             'status': 'error'
#         })

# if __name__ == '__main__':
#     app.run(debug=True, port=5001) 
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
import joblib
import os
import traceback

app = Flask(__name__)

# Load the model and transformers
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully")
            return model
        print(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        return None

def load_scaler():
    try:
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully")
            return scaler
        print(f"Scaler file not found at {scaler_path}")
        return None
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        print(traceback.format_exc())
        return None

def load_selector():
    try:
        selector_path = os.path.join(os.path.dirname(__file__), 'selector.pkl')
        if os.path.exists(selector_path):
            selector = joblib.load(selector_path)
            print("Selector loaded successfully")
            return selector
        print(f"Selector file not found at {selector_path}")
        return None
    except Exception as e:
        print(f"Error loading selector: {str(e)}")
        print(traceback.format_exc())
        return None

def load_pca():
    try:
        pca_path = os.path.join(os.path.dirname(__file__), 'pca.pkl')
        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)
            print("PCA loaded successfully")
            return pca
        print(f"PCA file not found at {pca_path}")
        return None
    except Exception as e:
        print(f"Error loading PCA: {str(e)}")
        print(traceback.format_exc())
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request")
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
        print("Features extracted successfully")
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        print(f"Features shape: {features.shape}")
        
        # Load model and transformers
        model = load_model()
        scaler = load_scaler()
        selector = load_selector()
        pca = load_pca()
        
        if model is None or scaler is None or selector is None or pca is None:
            error_msg = 'Model or transformers not found. Please ensure you have run main.py to train the model first.'
            print(error_msg)
            return jsonify({'error': error_msg, 'status': 'error'})
        
        # Transform the features in the correct order
        print("Applying feature transformations...")
        
        try:
            # First apply feature selection
            features_selected = selector.transform(features)
            print(f"After selection shape: {features_selected.shape}")
            
            # Then scale the selected features
            features_scaled = scaler.transform(features_selected)
            print(f"After scaling shape: {features_scaled.shape}")
            
            # Finally apply PCA
            features_pca = pca.transform(features_scaled)
            print(f"After PCA shape: {features_pca.shape}")
        except Exception as e:
            error_msg = f'Error during feature transformation: {str(e)}'
            print(error_msg)
            print(traceback.format_exc())
            return jsonify({'error': error_msg, 'status': 'error'})
        
        # Make prediction
        try:
            print("Making prediction...")
            prediction = model.predict(features_pca)[0]
            
            # Make sure to check the shape of the prediction probabilities
            probabilities = model.predict_proba(features_pca)[0]
            
            # Ensure we're getting the right probability (for class 1)
            if len(probabilities) == 2:
                probability = probabilities[1]  # Probability of class 1 (Parkinson's)
            else:
                probability = probabilities[0]  # In case there's only one probability value
                
            print(f"Prediction: {prediction}, Probability: {probability}")
            
            result = {
                'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
                'probability': f'{probability:.2%}',
                'status': 'success'
            }
            print("Prediction completed successfully")
            return jsonify(result)
        except Exception as e:
            error_msg = f'Error during model prediction: {str(e)}'
            print(error_msg)
            print(traceback.format_exc())
            return jsonify({'error': error_msg, 'status': 'error'})
    
    except Exception as e:
        error_msg = f'An error occurred during prediction: {str(e)}'
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            'error': error_msg,
            'status': 'error'
        })

if __name__ == '__main__':
    # Create the templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"Created templates directory at: {templates_dir}")
    
    # Check if index.html exists in templates directory
    index_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_path):
        print(f"Warning: index.html not found at {index_path}")
        # If index.html is in the current directory, copy it to templates
        current_dir_index = os.path.join(os.path.dirname(__file__), 'index.html')
        if os.path.exists(current_dir_index):
            import shutil
            shutil.copy(current_dir_index, index_path)
            print(f"Copied index.html from current directory to {index_path}")
    
    # Check for model files before starting the server
    print("Checking for model files...")
    model_exists = os.path.exists(os.path.join(os.path.dirname(__file__), 'model.pkl'))
    scaler_exists = os.path.exists(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))
    selector_exists = os.path.exists(os.path.join(os.path.dirname(__file__), 'selector.pkl'))
    pca_exists = os.path.exists(os.path.join(os.path.dirname(__file__), 'pca.pkl'))
    
    if not all([model_exists, scaler_exists, selector_exists, pca_exists]):
        print("Warning: Some model files are missing. Please run main.py first to train the model.")
        print(f"Model exists: {model_exists}")
        print(f"Scaler exists: {scaler_exists}")
        print(f"Selector exists: {selector_exists}")
        print(f"PCA exists: {pca_exists}")
    
    # Try multiple ports in case one is blocked
    ports = [8080, 8000, 5050, 3000]
    for port in ports:
        try:
            print(f"Attempting to start server on port {port}...")
            app.run(debug=True, port=port)
            break  # If successful, break out of the loop
        except Exception as e:
            print(f"Failed to start on port {port}: {str(e)}")
            if port == ports[-1]:
                print("All port attempts failed. Please try a different port manually.")