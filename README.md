# Dosh_model_prediction
🧠 AI-Powered Ayurveda Dosha Prediction System

A Machine Learning system that predicts a person’s Ayurvedic Dosha (Vata, Pitta, Kapha) using physiological, lifestyle, and symptom data. The model is optimized using GridSearchCV and provides confidence scores for each prediction.

This project demonstrates a complete end-to-end ML pipeline from preprocessing to deployment-ready inference.

📌 Features

✅ Predicts Dosha using Machine Learning

✅ Confidence score for each Dosha

✅ TF-IDF processing for symptom text

✅ One-Hot Encoding for categorical features

✅ Hyperparameter tuning using GridSearchCV

✅ Automatic best model selection

✅ Model saving and loading using Pickle

✅ Production-ready pipeline


🧬 Input Features

The model uses the following features:

Age

Gender

Prakriti

Symptoms

Stress Level

Sleep Pattern

Diet Type

Season

Climate


Machine Learning Pipeline

Dataset
 ↓
Data Cleaning
 ↓
Feature Encoding
   ├── TF-IDF (Symptoms)
   ├── OneHotEncoder (Categorical)
   └── Numeric Features (Age)
 ↓
Train-Test Split
 ↓
Model Comparison
 ↓
GridSearchCV Hyperparameter Optimization
 ↓
Best Model Selection
 ↓
Model Saving
 ↓
Prediction with Confidence Scores

🔧 Technologies Used

Python

Pandas

NumPy

Scikit-Learn

XGBoost

GridSearchCV

TF-IDF Vectorizer

Pickle

📊 Model Optimization

Hyperparameter tuning performed using GridSearchCV:
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
Features:

5-fold cross validation

Parallel processing (n_jobs = -1)

Automatic best model selection

##📈 Example Output
'predicted_dosha': 'Vata'

Confidence levels:
Kapha: 0.00%
Pitta: 0.00%
Vata: 100.00%

Final Output: Vata
