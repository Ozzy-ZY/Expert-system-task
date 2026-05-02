import numpy as np
from sklearn.neural_network import MLPClassifier

# Features (Symptoms): [Fever, Cough, Sneezing, Body Ache]
# 1 = Yes (Present), 0 = No (Absent)
X_train = np.array([
    [0, 1, 1, 0],  # Cold: Cough, Sneezing
    [1, 1, 0, 1],  # Flu: Fever, Cough, Body Ache
    [0, 0, 1, 0],  # Allergy: Sneezing only
    [0, 1, 1, 0],  # Cold
    [1, 1, 1, 1],  # Flu: All symptoms
    [0, 0, 1, 0],  # Allergy
    [1, 0, 0, 1],  # Flu: Fever, Body ache
    [0, 1, 0, 0],  # Cold: Just cough
])

# Labels (Diagnoses): 0 = Cold, 1 = Flu, 2 = Allergy
y_train = np.array([0, 1, 2, 0, 1, 2, 1, 0])

symptom_names = ["Fever", "Cough", "Sneezing", "Body Ache"]
diagnosis_map = {0: "Common Cold", 1: "Influenza (Flu)", 2: "Seasonal Allergies"}


# we use a Multi-Layer Perceptron (MLP) with a small hidden layer 
# suitable for this tiny dataset. We increase max_iter so it converges
print("Training the neural network expert system...")
nn_model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=42)
nn_model.fit(X_train, y_train)
print("Training complete.\n")


def expert_system_consultation(symptoms):
    """
    Takes a list of binary symptoms, queries the neural network, 
    and provides an expert recommendation.
    """
    # Convert input list to 2D numpy array for the model
    input_data = np.array([symptoms])
    
    # Get the prediction and the confidence (probability)
    prediction = nn_model.predict(input_data)[0]
    probabilities = nn_model.predict_proba(input_data)[0]
    confidence = probabilities[prediction] * 100
    
    # Format the output
    diagnosis = diagnosis_map[prediction]
    
    print("-" * 40)
    print("🩺 EXPERT SYSTEM CONSULTATION REPORT")
    print("-" * 40)
    print("Patient Symptoms reported:")
    for i, has_symptom in enumerate(symptoms):
        status = "Yes" if has_symptom else "No"
        print(f" - {symptom_names[i]}: {status}")
        
    print(f"\n>> System Diagnosis: {diagnosis}")
    print(f">> Confidence Level: {confidence:.2f}%\n")
    
    # Simple rule-based logic to supplement the neural network's decision
    if diagnosis == "Influenza (Flu)":
        print("Expert Advice: Please rest, drink plenty of fluids, and consider seeing a doctor if your fever persists.")
    elif diagnosis == "Common Cold":
        print("Expert Advice: Get some rest. Over-the-counter medications may help relieve symptoms.")
    elif diagnosis == "Seasonal Allergies":
        print("Expert Advice: Try an antihistamine and avoid known allergens if possible.")
    print("-" * 40 + "\n")



# test the expert system with a new patient who has a Fever and Body Ache, 
# but no Cough or Sneezing.
new_patient_symptoms = [1, 0, 0, 1] 
expert_system_consultation(new_patient_symptoms)

# Test with a patient who only has sneezing
another_patient = [0, 0, 1, 0]
expert_system_consultation(another_patient)
