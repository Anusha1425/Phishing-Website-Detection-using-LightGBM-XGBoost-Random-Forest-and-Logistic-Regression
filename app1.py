import pandas as pd
import numpy as np
import re

from urllib.parse import urlparse
from flask import Flask, request, render_template_string

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb


# -----------------------------
# 1 Load Dataset
# -----------------------------

df = pd.read_csv("url_dataset.csv")

df = df.dropna()

df['url'] = df['url'].astype(str)

df = df[df['url'].str.startswith('http')]


# -----------------------------
# 2 Convert Labels
# -----------------------------

df['type'] = df['type'].map({
    'legitimate': 0,
    'phishing': 1
})


# -----------------------------
# 3 Feature Extraction
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
# 4 Create Feature Matrix
# -----------------------------

X = df['url'].apply(extract_features)

X = pd.DataFrame(X.tolist())

y = df['type']

X = X.fillna(0)

X = X.replace([np.inf, -np.inf], 0)


# -----------------------------
# 5 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 6 Define Models
# -----------------------------

rf = RandomForestClassifier(n_estimators=200, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss'
)

lgb_model = lgb.LGBMClassifier(n_estimators=200)

lr = LogisticRegression(max_iter=1000)


# -----------------------------
# 7 Ensemble Model
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
# 8 Train Model
# -----------------------------

print("Training Model...")

ensemble_model.fit(X_train, y_train)


# -----------------------------
# 9 Flask Web App
# -----------------------------

app = Flask(__name__)

HTML_PAGE = """

<!DOCTYPE html>
<html>

<head>

<title>Phishing Detection</title>

<style>

body{
font-family: Arial;
background:#008080;
text-align:center;
padding-top:100px;
}

.container{
background:teal;
padding:40px;
width:500px;
margin:auto;
border-radius:10px;
box-shadow:0 0 10px rgba(0,0,0,0.2);
}

input{
width:80%;
padding:10px;
margin:10px;
}

button{
padding:10px 20px;
background:#007BFF;
color:white;
border:none;
border-radius:5px;
}

.result{
font-size:20px;
margin-top:20px;
}

</style>

</head>

<body>

<div class="container">

<h2>Phishing Website Detection</h2>

<form method="POST">

<input type="text" name="url" placeholder="Enter Website URL" required>

<br>

<button type="submit">Check URL</button>

</form>

{% if prediction %}

<div class="result">

{{prediction}}

</div>

{% endif %}

</div>

</body>

</html>
"""


# -----------------------------
# Prediction Function
# -----------------------------

def predict_url(url):

    features = extract_features(url)

    features = np.array(features).reshape(1, -1)

    prediction = ensemble_model.predict(features)[0]

    if prediction == 1:
        return "⚠️ Phishing Website Detected"
    else:
        return "✅ Legitimate Website"


# -----------------------------
# Flask Route
# -----------------------------

@app.route("/", methods=["GET","POST"])

def home():

    result = None

    if request.method == "POST":

        url = request.form["url"]

        result = predict_url(url)

    return render_template_string(HTML_PAGE, prediction=result)


# -----------------------------
# Run App
# -----------------------------

if __name__ == "__main__":

    app.run(debug=True)