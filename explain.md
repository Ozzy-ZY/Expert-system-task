# Neural Network Expert System: Project Documentation

## 1. Project Overview
Traditionally, Expert Systems (like MYCIN) relied on a hardcoded "Knowledge Base" of explicit `IF-THEN` rules crafted by human domain experts and an "Inference Engine" to evaluate them. 

This project demonstrates a modern approach: replacing the hardcoded rules with a **Machine Learning model** (specifically, an Artificial Neural Network). Instead of explicitly programming the medical rules, the network *learns* the patterns linking symptoms to diagnoses through a provided dataset.

## 2. System Architecture

### 2.1 The Knowledge Base (Dataset)
The system uses a mini-dataset acting as the "expert knowledge". 
* **Inputs (Features):** 4 binary symptoms `[Fever, Cough, Sneezing, Body Ache]`. `1` indicates presence, `0` indicates absence.
* **Outputs (Labels):** 3 discrete classes: `0` (Common Cold), `1` (Influenza), `2` (Seasonal Allergies).

### 2.2 The Inference Engine (Neural Network)
* **Model:** Multi-Layer Perceptron (MLP) Classifier using `scikit-learn`.
* **Architecture:** 4 input nodes (symptoms) $\rightarrow$ 1 hidden layer with 8 neurons $\rightarrow$ 3 output nodes (diagnoses).
* **Confidence Scoring:** Instead of a simple binary decision, the system uses the network's `predict_proba()` function to output a percentage-based confidence level for the diagnosis.

### 2.3 The User Interface
A terminal-based interface (`expert_system_consultation`) that accepts a patient's symptom array, transforms it for the model, retrieves the prediction and confidence score, and appends a hardcoded expert advice string based on the final prediction.

---

## 3. Potential Questions Answers

Here is a list of questions your Teaching Assistant (TA) might ask during a project discussion or defense, along with strong, technical answers.

### Q1: Why use a Neural Network instead of traditional IF-THEN rules for this?
**Answer:** Traditional rule-based systems suffer from the "knowledge acquisition bottleneck"—it is time-consuming to extract rules from experts, and the rules are hard to maintain as complexity grows. A Neural Network scales better. It can learn from raw historical data, handle fuzzy or incomplete inputs gracefully, and adapt to new data by simply retraining, rather than requiring a human to manually rewrite logical trees.

### Q2: Why did you choose an MLP with exactly `hidden_layer_sizes=(8,)`?
**Answer:** Model complexity should match data complexity. Our dataset is extremely small (4 features, 8 samples). Using a deep network with many layers or hundreds of neurons would lead to severe overfitting—the model would simply memorize the data rather than learn generalizable patterns. A single hidden layer with 8 neurons provides enough parameters to map the non-linear relationships between the 4 inputs and 3 outputs without being excessively heavy. 

### Q3: I see you used `max_iter=2000`. Isn't that high for such a small dataset?
**Answer:** Actually, small datasets sometimes struggle to converge quickly using standard gradient descent solvers like Adam (the default in `sklearn`'s `MLPClassifier`). Because the batch size is so small, the gradients can be noisy. Increasing the maximum iterations ensures the optimizer has enough epochs to reach the global minimum of the loss function and fully converge without throwing a `ConvergenceWarning`.

### Q4: How does the system handle uncertainty? What if the patient has confusing symptoms?
**Answer:** The system doesn't just output a deterministic class label. It uses the `predict_proba()` method, which applies a softmax function to the output layer of the neural network. This gives us a probability distribution across all three possible diagnoses. We extract the highest probability and present it as the system's "Confidence Level," providing transparency to the end-user.

### Q5: What are the main limitations of your current implementation?
**Answer:** 1. **Data Starvation:** The dataset is a "toy" dataset with only 8 rows. In a real-world scenario, deep learning models require thousands of records to generalize effectively.
2. **Feature Simplicity:** Symptoms are strictly boolean (Yes/No). Real medical data requires continuous variables (e.g., exact temperature, duration of cough in days).
3. **No Explainability:** Unlike decision trees or IF-THEN rules where you can trace the exact logic path, neural networks are "black boxes." A doctor using this system wouldn't know *why* the network made its decision, which is a significant issue in medical tech (addressed in modern AI via XAI techniques like SHAP).

### Q6: If you were to scale this up for a real hospital, what would you change in the backend architecture?
**Answer:** I would move away from an in-memory script. I would train the model and serialize it (e.g., using `joblib` or `ONNX`). Then, I would deploy it as an independent microservice using a framework like ASP.NET Core or FastAPI. The client applications would send JSON symptom payloads via REST or gRPC to this inference service, which would consult the model and return the probabilities. I would also use a proper database (like PostgreSQL) to log inference requests to monitor model drift over time.
