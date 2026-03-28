import pandas as pd
import numpy as np
import re

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb


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

def extract_features(url):

    try:

        parsed = urlparse(url)

        features = {}

        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphen'] = url.count('-')
        features['num_at'] = url.count('@')
        features['num_question'] = url.count('?')
        features['num_equal'] = url.count('=')
        features['num_and'] = url.count('&')
        features['num_percent'] = url.count('%')
        features['num_slash'] = url.count('/')
        features['num_www'] = url.count('www')

        features['num_http'] = url.count('http')
        features['num_https'] = url.count('https')

        features['path_length'] = len(parsed.path)

        features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

        return list(features.values())

    except:
        return [0]*13


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

    voting='hard'
)


# -----------------------------
# 10 Train Model
# -----------------------------

print("Training Model...")

ensemble_model.fit(X_train, y_train)


# -----------------------------
# 11 Prediction
# -----------------------------

y_pred = ensemble_model.predict(X_test)


# -----------------------------
# 12 Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


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