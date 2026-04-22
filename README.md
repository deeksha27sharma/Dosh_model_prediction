#🌿 AyurSage: Dosha & Disease Prediction SystemAn end-to-end Machine Learning pipeline that bridges ancient Ayurvedic wisdom with modern Data Science to provide personalized health insights.Built as part of a Final Year Project to predict Prakriti (Dosha), identify potential Ailments, and recommend Ayurvedic Treatments.

##🎯 What It Does

Provide your physiological attributes and symptoms. The system will:
Analyze — Process lifestyle and physical traits using an optimized ML pipeline.
Predict Dosha — Identify your dominant Dosha (Vata, Pitta, or Kapha).
Diagnose — Use the predicted Dosha + symptoms to identify potential diseases.
Prescribe — Automatically map findings to Diet, Exercise, and Ayurvedic Medicine.
Optimize — Use GridSearchCV to ensure the most accurate results possible.

##🏗️ ArchitectureUser Input (Symptoms/Lifestyle)
```
↓
Feature Engineering (TF-IDF + One-Hot Encoding)
↓
Stage 1: Dosha Classifier (Random Forest / XGBoost)
↓
Stage 2: Disease Diagnostic Model (Stage 1 Output + Inputs)
↓
Recommendation Engine (Curated Ayurveda Dataset)
↓
Final Health Report (Diet, Exercise, Medicine)
```
##🛠️ Tech StackComponent
TechnologyLanguagePython 3.xML 
FrameworksScikit-Learn, XGBoost
Data ProcessingPandas, NumPy
NLP TechniquesTF-IDF Vectorization
OptimizationGridSearchCV (Hyperparameter Tuning)
Model StoragePickle (.pkl)

##📁 Project StructureDosha_prediction_model/
```
├── Dataset/
│   ├── dosha_data.csv        # Raw Ayurvedic physiological data
│   └── treatment_map.csv     # Disease-to-treatment mapping
├── model/
│   ├── dosha_model.pkl       # Trained Dosha classifier
│   ├── disease_model.pkl     # Trained Disease classifier
│   └── encoders/             # TF-IDF & Label encoders
├── train.py                  # Training script for Dosha Model
├── train_disease_model.py    # Training script for Disease Model
├── predict_full_pipeline.py  # End-to-end inference script
├── requirements.txt          # Dependencies
└── README.md
```
##⚡ Installation1. 
1. Clone the repo
```
git clone https://github.com/deeksha27sharma/Dosha_prediction_model.git
cd Dosha_prediction_model
```
2. Install dependencies
 ```
pip install -r requirements.txt
```
3. Run Inference
To test the model with sample inputs:
```
python predict_full_pipeline.py
```
##🔑 Key Features

###Hybrid Inference — Uses a two-stage sequential model where the first prediction (Dosha) informs the second (Disease).
###Medical NLP — Successfully converts natural language symptom descriptions into numerical features via TF-IDF.
###Production Ready — Models are serialized and ready for integration into Web or Mobile APIs.
###Holistic Output — Goes beyond prediction to provide actionable lifestyle and dietary advice.
##📈 Model Performance

The models were evaluated using Stratified K-Fold Cross-Validation to handle class imbalances in Ayurvedic datasets:
Dosha Prediction Accuracy: ~99.8% (Near-perfect classification for Kapha, Pitta, and Vata classes).
Disease Identification: ~99% F1-Score using a chained Random Forest and Decision Tree pipeline.
Hyperparameters: Optimized via GridSearchCV for n_estimators and max_depth.

##Author: Diksha Sharma Final Year Computer Science Student
