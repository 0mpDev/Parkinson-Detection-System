import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib

# Load the dataset
df = pd.read_csv('parkinsons.csv')

# Exploratory Data Analysis
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['status'].value_counts())

# Visualizations
plt.figure(figsize=(15, 10))
sns.heatmap(df.drop('name', axis=1).corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Matrix")
plt.show()

# Feature and target separation
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Feature selection using ANOVA F-value
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("\nTop 10 important features:")
print(selected_features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Dimensionality reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"\nReduced to {pca.n_components_} components")

# Model Training and Evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return model, accuracy

# Random Forest
print("\nRandom Forest Classifier:")
rf_model, rf_acc = evaluate_model(RandomForestClassifier(random_state=42), 
                                 X_train_pca, X_test_pca, y_train, y_test)

# Save the model
joblib.dump(rf_model, 'model.pkl')

# Support Vector Machine
print("\nSupport Vector Machine:")
svm_model, svm_acc = evaluate_model(SVC(random_state=42), 
                                 X_train_pca, X_test_pca, y_train, y_test)

# Logistic Regression
print("\nLogistic Regression:")
lr_model, lr_acc = evaluate_model(LogisticRegression(random_state=42), 
                                 X_train_pca, X_test_pca, y_train, y_test)

# Feature Importance from Random Forest
if hasattr(rf_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X_train_pca.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train_pca.shape[1]), [f"PC-{i+1}" for i in range(X_train_pca.shape[1])], rotation=90)
    plt.tight_layout()
    plt.show()

# Compare model performances
models = ['Random Forest', 'SVM', 'Logistic Regression']
accuracies = [rf_acc, svm_acc, lr_acc]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()