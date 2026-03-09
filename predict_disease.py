import pickle
import pandas as pd

# Load models
dosha_model = pickle.load(open("model/best_dosha_model.pkl", "rb"))
dosha_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

disease_model = pickle.load(open("model/disease_model.pkl", "rb"))
disease_encoder = pickle.load(open("model/disease_label_encoder.pkl", "rb"))


def predict_disease_pipeline(input_dict):

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    # ---- DOSHA PREDICTION ----
    dosha_encoded = dosha_model.predict(input_df)
    predicted_dosha = dosha_encoder.inverse_transform(dosha_encoded)[0]

    dosha_probs = dosha_model.predict_proba(input_df)[0]

    dosha_probabilities = {
        dosha: float(prob)
        for dosha, prob in zip(dosha_encoder.classes_, dosha_probs)
    }

    # ---- DISEASE PREDICTION ----

    # Add predicted Dosha to input
    input_df["Dosha"] = predicted_dosha

    disease_encoded = disease_model.predict(input_df)
    predicted_disease = disease_encoder.inverse_transform(disease_encoded)[0]

    # ---- FINAL OUTPUT ----
    result = {
        "predicted_dosha": predicted_dosha,
        "dosha_probabilities": dosha_probabilities,
        "predicted_disease": predicted_disease
    }

    return result


# Example usage
sample = {
    "Age": 25,
    "Gender": "Female",
    "Prakriti": "Vata",
    "Symptoms": "dry skin, anxiety, constipation",
    "Stress Level": "High",
    "Sleep Pattern": "Insomnia",
    "Diet Type": "Vegetarian",
    "Season": "Winter",
    "Climate": "Cold"
}

output = predict_disease_pipeline(sample)

print(output)

sample2 = {
    "Age": 25,
    "Gender": "Female",
    "Prakriti": "kapha-Pitta",
    "Symptoms": "pale color, sluggish evacuation, coldness, heaviness",
    "Stress Level": "High",
    "Sleep Pattern": "Insomnia",
    "Diet Type": "Vegetarian",
    "Season": "Winter",
    "Climate": "Cold"
}
output2 = predict_disease_pipeline(sample2)

print(output2)

sample3 = {
    "Age": 25,
    "Gender": "Female",
    "Prakriti": "kapha-Pitta",
    "Symptoms": "skin eruption, pale or whitish color, oozing discharge",
    "Stress Level": "High",
    "Sleep Pattern": "Insomnia",
    "Diet Type": "Vegetarian",
    "Season": "Winter",
    "Climate": "Cold"
}
output3 = predict_disease_pipeline(sample3)

print(output3)

sample4 = {
    "Age": 25,
    "Gender": "Female",
    "Prakriti": "kapha-Pitta",
    "Symptoms": "dark red color, burning sensation",
    "Stress Level": "High",
    "Sleep Pattern": "Insomnia",
    "Diet Type": "Vegetarian",
    "Season": "Winter",
    "Climate": "Cold"
}
output4 = predict_disease_pipeline(sample4)

print(output4)

