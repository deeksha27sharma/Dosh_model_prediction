# 🧠 Ayurveda Dosha & Disease Prediction System

A production-ready Machine Learning system that predicts a person’s Ayurvedic Dosha (Vata, Pitta, Kapha), identifies potential diseases, and provides personalized treatment recommendations.

## 🚀 Key Features
* [cite_start]**Two-Stage Prediction Pipeline:** 1. **Dosha Classification:** Identifies primary Dosha with probabilistic confidence scores[cite: 5, 8].
    2. [cite_start]**Disease Diagnosis:** Uses the predicted Dosha along with lifestyle inputs to identify specific ailments[cite: 4, 5].
* [cite_start]**Full Treatment Lookup:** Automatically maps results to a curated dataset to provide Therapy, Medicine, Diet, and Exercise recommendations.
* [cite_start]**Advanced Feature Engineering:** * **TF-IDF Vectorization:** Processes natural language symptom descriptions.
    * [cite_start]**One-Hot Encoding:** Handles categorical lifestyle and climate data.
* [cite_start]**Optimized Models:** Hyperparameter tuning via GridSearchCV for Random Forest and XGBoost architectures.

## 🛠️ Tech Stack
* [cite_start]**Language:** Python [cite: 6]
* [cite_start]**ML Libraries:** Scikit-Learn, XGBoost, Pandas, NumPy 
* [cite_start]**Techniques:** TF-IDF, GridSearchCV, Pipeline, Stratified Train-Test Split 
* [cite_start]**Deployment:** Pickle (Serialized model storage) 

## 📊 Pipeline Architecture
1. [cite_start]**Preprocessing:** Handling missing values (median/mode) and encoding text/categorical features[cite: 6].
2. [cite_start]**Model 1 (Dosha):** Multi-class classification using an optimized Random Forest[cite: 6].
3. [cite_start]**Model 2 (Disease):** Sequential model that utilizes the output of Model 1 as a key feature[cite: 4, 7].
4. [cite_start]**Recommendation Engine:** Automated lookup of Ayurvedic treatments based on the identified Disease/Dosha pair.

## 📁 Project Structure
* [cite_start]`train.py`: Training and optimization for the Dosha model[cite: 6].
* [cite_start]`train_disease_model.py`: Training script for the disease classification model[cite: 7].
* [cite_start]`predict_full_pipeline.py`: Main inference script for end-to-end predictions.
* [cite_start]`model/`: Serialized `.pkl` files for models and label encoders.
