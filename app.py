"""
Heart Disease Prediction — Streamlit UI
Run: streamlit run app.py
Requires: heart_model.pkl and heart_scaler.pkl in same folder
"""

import streamlit as st
import numpy as np
import joblib
import json
import os

# ─── Page config ──────────────────────────────────────────────────────────────
import os
path = "heart_model.pkl"
st.caption(f"Model loaded: {os.path.getmtime(path)}")  # shows file timestamp

st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }

    .main { background: #0f1117; }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        color: #fff;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-low  { background: #0d2b1e; border: 2px solid #2ecc71; }
    .result-high { background: #2b0d0d; border: 2px solid #e74c3c; }
    .result-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.3rem; }
    .result-low  .result-title { color: #2ecc71; }
    .result-high .result-title { color: #e74c3c; }
    .result-prob { font-size: 3rem; font-weight: 700; }
    .result-low  .result-prob  { color: #2ecc71; }
    .result-high .result-prob  { color: #e74c3c; }
    .result-advice { color: #ccc; font-size: 0.9rem; margin-top: 0.5rem; }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 0;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    .metric-card {
        background: #1a1d27;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a2d3a;
    }
    .metric-label { color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #fff; font-size: 1.4rem; font-weight: 600; margin-top: 0.2rem; }

    div[data-testid="stSidebar"] { background: #16191f !important; }
    label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("heart_model.pkl"):
        return None, None, None
    model  = joblib.load("heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    info   = {}
    if os.path.exists("model_info.json"):
        with open("model_info.json") as f:
            info = json.load(f)
    return model, scaler, info

model, scaler, info = load_model()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">❤️ Heart Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter patient details below to assess heart disease risk</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️  `heart_model.pkl` not found!  "
             "Run the Colab notebook first, then place `heart_model.pkl` and "
             "`heart_scaler.pkl` in the same folder as this `app.py`.")
    st.stop()

# ─── Model stats banner ───────────────────────────────────────────────────────
if info:
    c1, c2, c3 = st.columns(3)
    for col, label, val in [
        (c1, "Model",    info.get("model_type", "RF").replace("Classifier", "")),
        (c2, "Accuracy", f"{info.get('accuracy', '—')}%"),
        (c3, "ROC-AUC",  str(info.get("roc_auc", "—"))),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ─── Input form ───────────────────────────────────────────────────────────────
st.markdown("### 🩺 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age      = st.slider("Age",           20, 80, 50)
    sex      = st.selectbox("Sex",        ["Female (0)", "Male (1)"])
    cp       = st.selectbox("Chest Pain Type",
                            ["Typical Angina (0)", "Atypical Angina (1)",
                             "Non-Anginal Pain (2)", "Asymptomatic (3)"])
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol     = st.slider("Serum Cholesterol (mg/dl)",     100, 600, 220)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
    restecg  = st.selectbox("Resting ECG",
                            ["Normal (0)", "ST-T abnormality (1)", "LV Hypertrophy (2)"])

with col2:
    thalach  = st.slider("Max Heart Rate Achieved",       50, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina",    ["No (0)", "Yes (1)"])
    oldpeak  = st.slider("ST Depression (Oldpeak)",       0.0, 6.5, 1.0, step=0.1)
    slope    = st.selectbox("Slope of Peak Exercise ST",
                            ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
    ca       = st.selectbox("Major Vessels Colored by Fluoroscopy",
                            ["0", "1", "2", "3"])
    thal     = st.selectbox("Thalassemia",
                            ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])


# ─── Helper: extract numeric value ────────────────────────────────────────────
def num(s): return int(s.split("(")[1].rstrip(")")) if "(" in str(s) else int(s)


# ─── Predict ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Analyse Heart Risk")

if predict_btn:
    features = np.array([[
        age,
        num(sex), num(cp),
        trestbps, chol,
        num(fbs), num(restecg),
        thalach,
        num(exang),
        oldpeak,
        num(slope), int(ca), num(thal)
    ]])

    scaled = scaler.transform(features)
    prob   = model.predict_proba(scaled)[0][1]   # probability of disease
    pred   = model.predict(scaled)[0]

    pct = round(prob * 100, 1)

    if pred == 1:
        st.markdown(f"""
        <div class="result-box result-high">
            <div class="result-title">⚠️  High Risk Detected</div>
            <div class="result-prob">{pct}%</div>
            <div class="result-advice">
                The model predicts a high likelihood of heart disease.<br>
                Please consult a cardiologist for a thorough evaluation.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-low">
            <div class="result-title">✅  Low Risk</div>
            <div class="result-prob">{pct}%</div>
            <div class="result-advice">
                The model predicts a low likelihood of heart disease.<br>
                Continue maintaining a healthy lifestyle!
            </div>
        </div>""", unsafe_allow_html=True)

    # Risk gauge
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Risk Probability Breakdown**")
    st.progress(prob)
    c1, c2 = st.columns(2)
    c1.metric("Disease Probability",    f"{pct}%")
    c2.metric("No Disease Probability", f"{round((1-prob)*100, 1)}%")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️  This tool is for educational purposes only. Not a substitute for medical advice.")

with st.expander("ℹ️  Feature Descriptions"):
    st.markdown("""
| Feature | Description |
|---------|-------------|
| **age** | Age of the patient |
| **sex** | 0 = Female, 1 = Male |
| **cp** | Chest pain type (0–3) |
| **trestbps** | Resting blood pressure (mmHg) |
| **chol** | Serum cholesterol (mg/dl) |
| **fbs** | Fasting blood sugar > 120 mg/dl (1 = true) |
| **restecg** | Resting ECG results (0–2) |
| **thalach** | Maximum heart rate achieved |
| **exang** | Exercise induced angina (1 = yes) |
| **oldpeak** | ST depression induced by exercise |
| **slope** | Slope of peak exercise ST segment (0–2) |
| **ca** | Number of major vessels (0–3) |
| **thal** | Thalassemia type (1–3) |
""")