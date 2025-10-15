import warnings

# This definitive filter will ignore any warning containing this specific text,
# no matter which scikit-learn file it comes from.
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")


# ---------- CONFIG ----------
# ... rest of your script

import os
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, log_loss, accuracy_score
import joblib

# pgmpy for Bayesian Network
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ---------- CONFIG ----------
DATA_PATH = "disease_symptoms.csv"
RANDOM_SEED = 42
MIN_SYMPTOM_FREQ = 5
TEST_SIZE = 0.2 # We will use 20% of the data for testing
TOP_K = 3

PERFECT_PREDICTORS_TO_EXCLUDE = [
    'acute liver failure', 'altered sensorium', 'anxiety', 'belly pain', 'blackheads',
    'bladder discomfort', 'blister', 'blood in sputum', 'bloody stool', 'brittle nails',
    'bruising', 'cold hands and feets', 'coma', 'congestion', 'continuous feel of urine',
    'cramps', 'dehydration', 'dischromic  patches', 'distention of abdomen',
    'drying and tingling lips', 'enlarged thyroid', 'extra marital contacts',
    'fluid overload', 'foul smell of urine', 'hip joint pain',
    'history of alcohol consumption', 'increased appetite', 'inflammatory nails',
    'internal itching', 'irregular sugar level', 'irritation in anus', 'knee pain',
    'lack of concentration', 'loss of smell', 'movement stiffness', 'mucoid sputum',
    'muscle wasting', 'nodal skin eruptions', 'pain behind the eyes',
    'pain during bowel movements', 'pain in anal region', 'palpitations',
    'passage of gases', 'patches in throat', 'polyuria', 'prominent veins on calf',
    'puffy face and eyes', 'pus filled pimples', 'receiving blood transfusion',
    'receiving unsterile injections', 'red sore around nose', 'redness of eyes',
    'runny nose', 'rusty sputum', 'scurring', 'shivering', 'silver like dusting',
    'sinus pressure', 'skin peeling', 'slurred speech', 'small dents in nails',
    'spinning movements', 'spotting  urination', 'stomach bleeding', 'sunken eyes',
    'swelling of stomach', 'swollen blood vessels', 'swollen extremeties', 'swollen legs',
    'throat irritation', 'toxic look (typhos)', 'ulcers on tongue', 'unsteadiness',
    'visual disturbances', 'watering from eyes', 'weakness in limbs',
    'weakness of one body side', 'weight gain', 'yellow crust ooze', 'yellow urine'
]

# ---------- UTIL: token normalization ----------
def normalize_token(s):
    if pd.isna(s): return None
    s = str(s).lower().strip()
    s = re.sub(r'[_/\\]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s-]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s if s else None

SYN_MAP = {
    'skin rash': 'skin_rash', 'rash': 'skin_rash', 'feverish': 'fever',
    'high fever': 'fever', 'runny nose': 'runny_nose'
}

def apply_synonyms(tok):
    return SYN_MAP.get(tok, tok) if tok else None

# ---------- LOAD & PREPROCESS ----------
# In train.py

# Replace the old function with this one

def load_and_preprocess(path=DATA_PATH):
    raw = pd.read_csv(path)
    
    # This part is correct!
    print(f"Dataset shape before dropping duplicates: {raw.shape}")
    raw.drop_duplicates(inplace=True)
    print(f"Dataset shape after dropping duplicates: {raw.shape}")

    if 'Disease' not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: 'Disease'})
    disease_col = 'Disease'
    raw[disease_col] = raw[disease_col].apply(normalize_token) # Clean disease names
    symptom_cols = [c for c in raw.columns if c != disease_col]
    
    def norm_row_tokens(row):
        toks = [apply_synonyms(normalize_token(row.get(c))) for c in symptom_cols]
        return [t for t in toks if t]
        
    raw['symptom_tokens'] = raw.apply(norm_row_tokens, axis=1)
    raw = raw.dropna(subset=[disease_col])
    raw = raw[raw['symptom_tokens'].apply(len) > 0].reset_index(drop=True)
    return raw, disease_col

# Replace the old function with this one

def tokens_to_multihot(df, freq_thresh=MIN_SYMPTOM_FREQ, exclude_list=None):
    all_tokens = Counter(t for toks in df['symptom_tokens'] for t in set(toks))
    keep_tokens = sorted([t for t, c in all_tokens.items() if c >= freq_thresh])
    
    if exclude_list:
        original_count = len(keep_tokens)
        keep_tokens = [tok for tok in keep_tokens if tok not in exclude_list]
        print(f"Excluded {original_count - len(keep_tokens)} perfect predictor symptoms.")

    print(f"Number of symptom features being used: {len(keep_tokens)}")
    
    # This is the part that was missing
    features = np.zeros((len(df), len(keep_tokens)), dtype=int)
    tok_to_idx = {t: i for i, t in enumerate(keep_tokens)}
    for i, toks in enumerate(df['symptom_tokens']):
        for t in set(toks):
            if t in tok_to_idx:
                features[i, tok_to_idx[t]] = 1
    
    return pd.DataFrame(features, columns=keep_tokens), keep_tokens

# ---------- MODELS ----------
def train_bernoulli_nb(X_train, y_train):
    base = BernoulliNB()
    clf = CalibratedClassifierCV(base, method='isotonic', cv=3)
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_SEED)
    clf = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    clf.fit(X_train, y_train)
    return clf

def train_bayesian_network(df_with_targets, disease_col, symptom_feature_names):
    edges = [(disease_col, s) for s in symptom_feature_names]
    model = DiscreteBayesianNetwork(edges)
    model.fit(df_with_targets, estimator=BayesianEstimator, equivalent_sample_size=10)
    return model

# ---------- EVALUATION HELPERS ----------
def top_k_accuracy(probs, y_true_indices, k=TOP_K):
    top_k_preds = np.argsort(-probs, axis=1)[:, :k]
    matches = [y_true_indices[i] in top_k_preds[i] for i in range(len(y_true_indices))]
    return np.mean(matches)

def evaluate_classifier(clf, X, y, le):
    """Evaluates scikit-learn classifiers."""
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    return {
        'accuracy': accuracy_score(y, preds),
        'top_3_accuracy': top_k_accuracy(probs, y, k=TOP_K),
        'macro_f1': f1_score(y, preds, average='macro', zero_division=0),
        'log_loss': log_loss(y, probs)
    }

def evaluate_bn(bn_model, X, y_str, feature_names, le):
    """Evaluates the Bayesian Network model. (Can be slow)"""
    disease_col = 'Disease' # As used in training
    classes = le.classes_
    bn_probs = []
    
    infer = VariableElimination(bn_model)
    for i in range(X.shape[0]):
        evidence = {feat: val for feat, val in zip(feature_names, X[i])}
        prob_dist = infer.query(variables=[disease_col], evidence=evidence, show_progress=False)
        prob_map = {state: val for state, val in zip(prob_dist.state_names[disease_col], prob_dist.values)}
        probs_arr = np.array([prob_map.get(cls, 0.0) for cls in classes])
        bn_probs.append(probs_arr)
        
    bn_probs = np.vstack(bn_probs)
    y_numeric = le.transform(y_str)
    bn_preds_numeric = np.argmax(bn_probs, axis=1)

    return {
        'accuracy': accuracy_score(y_numeric, bn_preds_numeric),
        'top_3_accuracy': top_k_accuracy(bn_probs, y_numeric, k=TOP_K),
        'macro_f1': f1_score(y_numeric, bn_preds_numeric, average='macro', zero_division=0),
        'log_loss': log_loss(y_numeric, bn_probs)
    }

# ---------- MAIN TRAINING & EVALUATION PIPELINE ----------
def main():
    print("Loading and preprocessing data...")
    raw, disease_col = load_and_preprocess(DATA_PATH)
    # Update this line to pass the exclude list
    feat_df, feature_names = tokens_to_multihot(raw, freq_thresh=MIN_SYMPTOM_FREQ, exclude_list=PERFECT_PREDICTORS_TO_EXCLUDE)

    le = LabelEncoder()
    y = le.fit_transform(raw[disease_col])
    print(f"Number of diseases: {len(le.classes_)}")

    X = feat_df.values
    # **IMPORTANT: We now keep the test set for evaluation**
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")

    # --- Model Training ---
    print("\n--- Training Models ---")
    print("Training BernoulliNB...")
    nb_clf = train_bernoulli_nb(X_train, y_train)

    print("Training Random Forest...")
    rf_clf = train_random_forest(X_train, y_train)

    df_for_bn = pd.DataFrame(X_train, columns=feature_names)
    df_for_bn[disease_col] = le.inverse_transform(y_train)
    print("Training Bayesian Network...")
    bn_model = train_bayesian_network(df_for_bn, disease_col, feature_names)

    # --- Model Saving ---
    print("\n--- Saving Models ---")
    joblib.dump(nb_clf, "model_bernoulli_nb.joblib")
    joblib.dump(rf_clf, "model_rf.joblib")
    with open("model_bn.pkl", "wb") as f: pickle.dump(bn_model, f)
    with open("label_encoder.pkl", "wb") as f: pickle.dump(le, f)
    with open("feature_names.pkl", "wb") as f: pickle.dump(feature_names, f)
    print("All models and artifacts saved.")

    # --- Model Evaluation ---
    print("\n--- Evaluating Models ---")
    
    # Evaluate Bernoulli NB
    nb_train_eval = evaluate_classifier(nb_clf, X_train, y_train, le)
    nb_test_eval = evaluate_classifier(nb_clf, X_test, y_test, le)
    print("\n--- Bernoulli Naive Bayes Results ---")
    print(f"Training Set:   Accuracy={nb_train_eval['accuracy']:.2%}, Top-3 Acc={nb_train_eval['top_3_accuracy']:.2%}, F1-Score={nb_train_eval['macro_f1']:.3f}")
    print(f"Testing Set:    Accuracy={nb_test_eval['accuracy']:.2%}, Top-3 Acc={nb_test_eval['top_3_accuracy']:.2%}, F1-Score={nb_test_eval['macro_f1']:.3f}")

    # Evaluate Random Forest
    rf_train_eval = evaluate_classifier(rf_clf, X_train, y_train, le)
    rf_test_eval = evaluate_classifier(rf_clf, X_test, y_test, le)

    print("\n--- Bayesian Network Results ---")
    print(f"Training Set:   Accuracy={rf_train_eval['accuracy']:.2%}, Top-3 Acc={rf_train_eval['top_3_accuracy']:.2%}, F1-Score={rf_train_eval['macro_f1']:.3f}")
    print(f"Testing Set:    Accuracy={rf_test_eval['accuracy']:.2%}, Top-3 Acc={rf_test_eval['top_3_accuracy']:.2%}, F1-Score={rf_test_eval['macro_f1']:.3f}")

    # Evaluate Bayesian Network
    
    y_train_str = le.inverse_transform(y_train)
    y_test_str = le.inverse_transform(y_test)
    bn_train_eval = evaluate_bn(bn_model, X_train, y_train_str, feature_names, le)
    bn_test_eval = evaluate_bn(bn_model, X_test, y_test_str, feature_names, le)
    print(f"Training Set:   Accuracy={bn_train_eval['accuracy']:.2%}, Top-3 Acc={bn_train_eval['top_3_accuracy']:.2%}, F1-Score={bn_train_eval['macro_f1']:.3f}")
    print(f"Testing Set:    Accuracy={bn_test_eval['accuracy']:.2%}, Top-3 Acc={bn_test_eval['top_3_accuracy']:.2%}, F1-Score={bn_test_eval['macro_f1']:.3f}")
    print("\nEvaluation complete. You can run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()