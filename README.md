# Parkinson's Disease Prediction System

A machine learning-based web application for predicting Parkinson's disease using voice measurements. This project uses various voice features to predict the likelihood of Parkinson's disease in patients.

## Features

- Machine Learning Models:
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Logistic Regression
- Web Interface for easy prediction
- Feature importance visualization
- Model performance comparison
- Real-time predictions

## Project Structure

```
parkinson_project/
├── app.py              # Flask web application
├── main.py            # Model training and evaluation
├── requirements.txt   # Project dependencies
├── model.pkl         # Trained model
├── scaler.pkl        # Feature scaler
├── parkinsons.csv    # Dataset
└── templates/
    └── index.html    # Web interface template
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/parkinson_project.git
cd parkinson_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python main.py
```

4. Run the web application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5001
```

## Dataset

The project uses the Parkinson's Disease dataset which contains various voice measurements from patients. The dataset includes 23 voice features and a binary target variable indicating the presence of Parkinson's disease.

## Model Performance

- Random Forest Classifier: 89.74% accuracy
- Support Vector Machine: 92.31% accuracy
- Logistic Regression: 87.18% accuracy

## Usage

1. Enter the voice measurements in the web interface
2. Click the "Predict" button
3. View the prediction result and probability

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn team for the machine learning library
- Flask team for the web framework 