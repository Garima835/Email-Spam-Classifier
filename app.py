from flask import Flask, render_template, request
import joblib
import string
import numpy as np
import json
import time
import re
from urllib.parse import urlparse
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# ---------------- CONFIGURATION ----------------

app = Flask(__name__)

GENAI_API_KEY = "AIzaSyDjsvXz2vEwJFMC33FJvxSIiHRpHwD2Efo"
genai.configure(api_key=GENAI_API_KEY)

model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

last_api_call = 0


# ---------------- TEXT CLEANING ----------------

def clean_text(text):

    text = "".join([char for char in text if char not in string.punctuation])

    words = [word.lower() for word in text.split() if word.lower() not in stop_words]

    stemmed_words = [stemmer.stem(word) for word in words]

    return " ".join(stemmed_words)


# ---------------- SPAM KEYWORD INDICATORS ----------------

def get_spam_indicators(text):

    cleaned = clean_text(text)

    vec = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()

    spam_log_probs = model.feature_log_prob_[1]

    indicators = []

    for word in cleaned.split():

        if word in feature_names:

            idx = np.where(feature_names == word)[0][0]

            importance = spam_log_probs[idx]

            indicators.append((word, importance))

    indicators.sort(key=lambda x: x[1], reverse=True)

    return [word for word, imp in indicators[:5]]


# ---------------- PHISHING WORD DETECTION ----------------

def detect_phishing(text):

    phishing_dict = {
        "verify": "Requests account verification",
        "password": "Possible credential theft attempt",
        "bank": "Financial information request",
        "login": "Login credential harvesting",
        "urgent": "Creates urgency to force action",
        "otp": "One-time password request",
        "account": "Account access request",
        "click": "External link redirection",
        "reset": "Password reset attempt",
        "security": "Fake security warning"
    }

    found = []

    lower_text = text.lower()

    for word, reason in phishing_dict.items():

        if word in lower_text:

            found.append({
                "word": word,
                "reason": reason
            })

    return found


# ---------------- URL PHISHING DETECTION ----------------

def detect_urls(text):

    url_pattern = r'(https?://[^\s]+)'

    urls = re.findall(url_pattern, text)

    suspicious = []

    for url in urls:

        domain = urlparse(url).netloc

        if len(domain) > 25 or "-" in domain:

            suspicious.append({
                "url": url,
                "reason": "Suspicious or misleading domain structure"
            })

    return suspicious


# ---------------- GEMINI ANALYSIS ----------------

def analyze_with_gemini(text):

    global last_api_call

    try:

        if time.time() - last_api_call < 5:

            return {
                "verdict": "Skipped",
                "analysis": "Rate limit protection active",
                "reply": "None"
            }

        last_api_call = time.time()

        prompt = f"""
You are a cybersecurity AI.

Analyze this email:

{text}

Return JSON:

{{
"verdict": "Spam or Ham",
"analysis": "short explanation",
"reply": "reply if ham otherwise None"
}}
"""

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)

        json_text = response.text.replace("```json", "").replace("```", "").strip()

        return json.loads(json_text)

    except Exception as e:

        print("Gemini Error:", e)

        return {
            "verdict": "Error",
            "analysis": str(e),
            "reply": "None"
        }


# ---------------- LOCAL AI EXPLANATION ----------------

def explain_spam(nb_prediction, indicators, phishing_flags, url_flags):

    if nb_prediction == "spam":

        reasons = []

        if indicators:
            reasons.append("Spam keywords detected")

        if phishing_flags:
            reasons.append("Phishing language found")

        if url_flags:
            reasons.append("Suspicious URLs detected")

        if not reasons:
            reasons.append("Suspicious message pattern")

        return ", ".join(reasons)

    else:

        return "No suspicious patterns detected"


# ---------------- ROUTES ----------------

@app.route('/')
def home():

    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form.get('message')

    if not message:

        return render_template('index.html', prediction="Error")

    cleaned_message = clean_text(message)

    data = vectorizer.transform([cleaned_message])

    nb_prediction = model.predict(data)[0]

    nb_prob = model.predict_proba(data).max() * 100

    indicators = get_spam_indicators(message)

    phishing_flags = detect_phishing(message)

    url_flags = detect_urls(message)

    explanation = explain_spam(nb_prediction, indicators, phishing_flags, url_flags)

    if nb_prob < 85:

        gemini_result = analyze_with_gemini(message)

    else:

        gemini_result = {
            "verdict": nb_prediction,
            "analysis": "High confidence local model",
            "reply": "None"
        }

    if gemini_result["verdict"] == "Error":

        final_verdict = nb_prediction

    elif gemini_result["verdict"].lower() == nb_prediction.lower():

        final_verdict = nb_prediction

    else:

        final_verdict = gemini_result["verdict"]

    risk_score = 0

    if nb_prediction == "spam":
        risk_score += 40

    if gemini_result["verdict"].lower() == "spam":
        risk_score += 40

    risk_score += len(indicators) * 3

    risk_score += len(phishing_flags) * 5

    risk_score += len(url_flags) * 10

    risk_score = min(risk_score, 100)

    return render_template(
        'index.html',
        prediction=nb_prediction,
        confidence=f"{nb_prob:.2f}%",
        message=message,
        indicators=indicators,
        phishing_flags=phishing_flags,
        url_flags=url_flags,
        gemini_verdict=gemini_result["verdict"],
        gemini_analysis=gemini_result["analysis"],
        gemini_reply=gemini_result["reply"],
        final_verdict=final_verdict,
        risk_score=risk_score,
        explanation=explanation
    )


if __name__ == '__main__':

    app.run(debug=True)