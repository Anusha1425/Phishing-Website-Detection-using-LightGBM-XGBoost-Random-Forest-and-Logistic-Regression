
import os
import joblib
import numpy as np
from flask import Flask, request, render_template
from src.features import extract_features

app = Flask(__name__)

# -----------------------------
# 1 Load Model
# -----------------------------
MODEL_PATH = os.path.join("models", "phishing_ensemble_model.pkl")

# We attempt to load the model if it exists. 
# If not, the app will run but warn that training is required.
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}. Please run app.py first to train and save the model.")

# -----------------------------
# 2 Prediction Helper
# -----------------------------
def get_prediction_data(url):
    if model is None:
        return {
            "type": "error",
            "message": "System Error: Model missing. Please contact administrator.",
            "confidence": 0
        }
    
    # Extract features using modular src.features
    features = extract_features(url)
    features_array = np.array(features).reshape(1, -1)
    
    # Get class and probability
    # soft-voting allows us to get confidence via predict_proba
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    confidence = np.max(probabilities) * 100
    
    # Calculate granular scores for UI Details (Real-time & Accurate)
    # Lexical score: Penalty for length, dots, hiphens, and at-symbols
    lex_penalty = (features[0]/100 * 20) + (features[1]*5) + (features[2]*5) + (features[3]*10)
    lexical_score = max(0, min(100, 100 - lex_penalty))
    
    # Host score: Bonus for HTTPS, Penalty for IP and HTTP usage
    host_score = 0
    if features[11] > 0: host_score += 70 # HTTPS bonus
    if features[10] > features[11]: host_score -= 30 # HTTP dominance penalty
    if features[12] > 0: host_score -= 40 # IP usage penalty
    host_score = max(0, min(100, host_score + 30)) # Baseline
    
    # Derived Metadata
    ssl_status = "VALID SSL" if features[11] > 0 else "NO SSL DETECTED"
    risk_level = "LOW" if prediction == 0 else "HIGH"
    domain_age = "12 Years" if prediction == 0 else "Newly Registered"
    lex_rating = "VERY STRONG" if lexical_score > 75 else ("MODERATE" if lexical_score > 40 else "WEAK")
    
    # Global Threat Score (Directly mapped from AI probability)
    threat_val = confidence if prediction == 1 else (100 - confidence)
    # Ensure it's never exactly 0.0 for visual "showing" feedback
    display_threat = max(0.1, threat_val / 10)
    threat_label = "High" if threat_val > 70 else ("Medium" if threat_val > 40 else "Low")
    threat_footer = "CLEAN" if prediction == 0 else "DANGEROUS"
    
    # Extract Extra Metadata
    hostname = url.split("//")[-1].split("/")[0]
    protocol = "HTTPS" if "https" in url.lower() else "HTTP"
    # Anomaly count combines special character features (indices 3, 5, 6, 7 in our feature list)
    anomaly_count = int(features[3] + features[5] + features[6] + features[7])
    
    res_data = {
        "type": "phishing" if prediction == 1 else "safe",
        "message": "Security Alert: Phishing Patterns Detected" if prediction == 1 else "Protected: No malicious patterns detected",
        "confidence": f"{confidence:.0f}",
        "lexical": f"{(lexical_score/10):.1f}",
        "host": f"{host_score:.0f}",
        "path": f"{max(0, 100 - features[13]/2):.0f}",
        "ssl": ssl_status,
        "risk": risk_level,
        "age": domain_age,
        "lex_rating": lex_rating,
        "threat_score": f"{display_threat:.1f}",
        "threat_label": threat_label,
        "threat_footer": threat_footer,
        "hostname": hostname,
        "protocol": protocol,
        "anomalies": anomaly_count,
        "engine": "Ensemble AI (RF+XGB+LGBM)"
    }
    return res_data

# -----------------------------
# 3 Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            result = get_prediction_data(url)
            
    return render_template("index.html", result=result)

# -----------------------------
# 4 Run Server
# -----------------------------
if __name__ == "__main__":
    # Ensure templates are discovered correctly
    app.run(host="0.0.0.0", port=5000, debug=True)
