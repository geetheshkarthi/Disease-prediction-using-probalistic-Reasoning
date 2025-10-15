import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pgmpy.inference import VariableElimination

# ---------- CONFIG & CONSTANTS ----------
DISEASE_COL = 'Disease'
TOP_K = 5

# Helper function for BN prediction
def bn_predict_proba(bn_model, evidence_dict, disease_col, disease_values):
    infer = VariableElimination(bn_model)
    result = infer.query(variables=[disease_col], evidence=evidence_dict, show_progress=False)
    prob_map = {state: float(val) for state, val in zip(result.state_names[disease_col], result.values)}
    # Ensure all diseases have a probability, even if it's 0
    full_probs = {disease: prob_map.get(disease, 0.0) for disease in disease_values}
    return full_probs

# Function to format predictions into a nice table
def format_predictions(prob_array, classes, k):
    idx = np.argsort(-prob_array)[:k]
    preds = [(classes[i], f"{prob_array[i]:.2%}") for i in idx]
    return pd.DataFrame(preds, columns=['Predicted Disease', 'Probability'])

# Use Streamlit's cache to load models only once
@st.cache_resource
def load_models():
    """Load all models and necessary artifacts from disk."""
    try:
        nb_model = joblib.load("model_bernoulli_nb.joblib")
        rf_model = joblib.load("model_rf.joblib")
        with open("model_bn.pkl", "rb") as f:
            bn_model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return nb_model, rf_model, bn_model, le, feature_names
    except FileNotFoundError:
        st.error(
            "Model files not found. Please run `train.py` first to train the models and generate the necessary files."
        )
        return None, None, None, None, None

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🩺 Disease Prediction from Symptoms")
st.markdown(
    "Select the symptoms you are experiencing from the dropdown below, and the models will predict the most likely diseases."
)

# Load models
nb_model, rf_model, bn_model, le, feature_names = load_models()

# Only proceed if models were loaded successfully
if all([nb_model, rf_model, bn_model, le, feature_names]):
    # UI for symptom selection
    selected_symptoms = st.multiselect(
        "Select your symptoms:",
        options=sorted(feature_names),
        placeholder="Type to search for symptoms..."
    )

    if st.button("Predict Disease", type="primary"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Create feature vector for scikit-learn models
            vec = np.zeros((1, len(feature_names)), dtype=int)
            for symptom in selected_symptoms:
                if symptom in feature_names:
                    idx = feature_names.index(symptom)
                    vec[0, idx] = 1

            # --- PREDICTIONS ---
            # 1. Bernoulli Naive Bayes
            nb_proba = nb_model.predict_proba(vec)[0]
            
            # 2. Random Forest
            rf_proba = rf_model.predict_proba(vec)[0]
            
            # 3. Bayesian Network
            evidence = {feat: 1 if feat in selected_symptoms else 0 for feat in feature_names}
            classes = le.classes_
            bn_prob_map = bn_predict_proba(bn_model, evidence, DISEASE_COL, classes)
            bn_probs_arr = np.array([bn_prob_map.get(c, 0.0) for c in classes])

            # --- DISPLAY RESULTS (Correctly indented) ---
            st.subheader("Top Predictions")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Bernoulli Naive Bayes")
                st.dataframe(format_predictions(nb_proba, classes, TOP_K), width='stretch', hide_index=True)

            with col2:
                st.markdown("#### Random Forest")
                st.dataframe(format_predictions(rf_proba, classes, TOP_K), width='stretch', hide_index=True)

            with col3:
                st.markdown("#### Bayesian Network")
                st.dataframe(format_predictions(bn_probs_arr, classes, TOP_K), width='stretch', hide_index=True)
