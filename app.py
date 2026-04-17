import pandas as pd
import numpy as np
import re

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from src.features import extract_features, get_feature_names


# -----------------------------
# 1 Load Dataset
# -----------------------------

df = pd.read_csv("url_dataset.csv")

print("Original Dataset Shape:", df.shape)


# -----------------------------
# 2 Clean Dataset
# -----------------------------

df = df.dropna()

df['url'] = df['url'].astype(str)

df = df[df['url'].str.startswith('http')]

print("Cleaned Dataset Shape:", df.shape)


# -----------------------------
# 3 Convert Labels
# -----------------------------

df['type'] = df['type'].map({
    'legitimate': 0,
    'phishing': 1
})


# -----------------------------
# 4 Feature Extraction
# -----------------------------

# Use modular features from src/features.py
# extract_features is now imported from src.features


# -----------------------------
# 5 Create Feature Matrix
# -----------------------------

X = df['url'].apply(extract_features)

X = pd.DataFrame(X.tolist())

y = df['type']


# -----------------------------
# 6 Fix NaN or Infinite Values
# -----------------------------

X = X.fillna(0)

X = X.replace([np.inf, -np.inf], 0)


print("Feature Matrix Shape:", X.shape)


# -----------------------------
# 7 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 8 Define Models
# -----------------------------

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss'
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=200
)

lr = LogisticRegression(
    max_iter=1000
)


# -----------------------------
# 9 Ensemble Model
# -----------------------------

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('lr', lr)
    ],
    voting='soft'
)

# -----------------------------
# 10 Train & Evaluate Models
# -----------------------------

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "Voting Classifier (Ours)": ensemble_model
}

results = []

print("Starting Model Training and Evaluation...")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": f"{acc*100:.1f}%",
        "Precision": f"{prec*100:.1f}%",
        "Recall": f"{rec*100:.1f}%",
        "F1-Score": f"{f1*100:.1f}%",
        "ROC-AUC": f"{auc:.3f}"
    })

# -----------------------------
# 11 Display Results Table
# -----------------------------

results_df = pd.DataFrame(results)
print("\n--- COMPARATIVE PERFORMANCE METRICS ---")
print(results_df.to_string(index=False))

# -----------------------------
# 12 Save Results & Models
# -----------------------------
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nResults exported to model_comparison_results.csv")

# Save the final ensemble model for web deployment
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "phishing_ensemble_model.pkl")
joblib.dump(ensemble_model, model_path)
print(f"Ensemble model saved to {model_path}")


# -----------------------------
# 13 Predict New URL
# -----------------------------

def predict_url(url):

    features = extract_features(url)

    features = np.array(features).reshape(1, -1)

    prediction = ensemble_model.predict(features)[0]

    if prediction == 1:
        print(" Phishing Website Detected")
    else:
        print(" Legitimate Website")