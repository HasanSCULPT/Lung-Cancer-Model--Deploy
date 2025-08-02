
# 📁 Folder Structure:
# Lung-Cancer-Model-Deploy/
# ├─ lung_cancer_app.py
# ├─ lung_cancer_pipeline.pkl
# ├─ logo.png
# ├─ feathered_bg.png   ✅ ← background image 
# ├─ feathered_bg.png
# └─ requirements.txt


# =======================================
# File: lung_cancer_app.py
# =======================================


# =========================================================
# ✅ Lung Cancer Diagnostic App (Streamlit)
# By HasanSCULPT | DSA 2025
# =========================================================
# 🔹 Deployment: Streamlit Cloud or Local
# 🔹 Features:
#    ✅ Multilingual UI (EN, FR, AR, RU, UK)
#    ✅ Upload CSV for batch prediction (with cleaning)
#    ✅ Individual prediction form
#    ✅ Threshold tuning (Max Recall & ROC)
#    ✅ SHAP (KernelExplainer) OR Permutation toggle
#    ✅ Confidence bar chart
#    ✅ Download results as CSV & PDF
#    ✅ Email sending (placeholders included)
#    ✅ Background image & logo supported
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
import smtplib
from email.message import EmailMessage
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve
from fpdf import FPDF

# ✅ CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="wide")

# ====================================
# ✅ A. BACKGROUND OPTION 1 (JPEG with st.markdown)
# ====================================
def set_background_jpeg(jpg_file):
    with open(jpg_file, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Uncomment to activate JPEG background
# set_background_jpeg("background.jpg")

# ====================================
# ✅ B. BACKGROUND OPTION 2 (PNG with st.cache_data)
# ====================================
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
# Uncomment to activate PNG background
# set_png_as_page_bg("background.png")

# ----------------------------
# ✅ Load Model & Features
# ----------------------------
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

expected_features = [
    "AGE", "GENDER", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN",
    "SYMPTOM_SCORE", "LIFESTYLE_SCORE"
]

importance_data = {
    "Feature": ["SYMPTOM_SCORE", "LIFESTYLE_SCORE", "SHORTNESS OF BREATH", "COUGHING", "ANXIETY"],
    "Importance": [0.0629, 0.0370, 0.0274, 0.0258, 0.0241]
}

# ----------------------------
# ✅ THEME TOGGLE
# ----------------------------
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body{background-color:#121212;color:white}</style>", unsafe_allow_html=True)

# ----------------------------
# ✅ HEADER
# ----------------------------
st.title("🔬 Lung Cancer Diagnostics Centre")
st.write("### By HasanSCULPT | DSA 2025")

# ----------------------------
# ✅ LAYOUT: TWO COLUMNS
# ----------------------------
col1, col2 = st.columns(2)

# ----------------------------
# ✅ LEFT PANEL: BATCH PREDICTION
# ----------------------------
with col1:
    st.subheader("📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    threshold_manual = st.slider("🎛 Manual Threshold", 0.0, 1.0, 0.5, 0.01)

    suggested_threshold = None
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df_input.head())

        # Ensure all required columns exist
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_names]

        # Predict
        proba = pipeline.predict_proba(df_input)[:, 1]
        preds = (proba > threshold_manual).astype(int)

        # Suggested threshold
        fpr, tpr, thresholds = roc_curve(np.zeros_like(proba), proba)
        youden = tpr - fpr
        suggested_threshold = thresholds[np.argmax(youden)]

        st.info(f"🔍 Suggested Threshold (ROC): {suggested_threshold:.2f}")

        # Show results
        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = preds
        st.write("### Prediction Results")
        st.dataframe(df_output)

        # Download CSV
        st.download_button("📥 Download Results CSV", df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        # Probability histogram
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, color="skyblue", edgecolor="black")
        ax.axvline(threshold_manual, color='red', linestyle='--', label='Manual Threshold')
        ax.axvline(suggested_threshold, color='green', linestyle='--', label='Suggested')
        ax.legend()
        st.pyplot(fig)

# ----------------------------
# ✅ RIGHT PANEL: INDIVIDUAL PREDICTION
# ----------------------------
with col2:
    st.subheader("👤 Individual Prediction")

    st.write("---")
    age = st.number_input("Age", 0, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", [0, 1])
    anxiety = st.selectbox("Anxiety", [0, 1])
    alcohol = st.selectbox("Alcohol Consuming", [0, 1])
    peer_pressure = st.selectbox("Peer Pressure", [0, 1])
    cough = st.selectbox("Coughing", [0, 1])
    short_breath = st.selectbox("Shortness of Breath", [0, 1])
    yellow_fingers = st.selectbox("yellow fingers", [0, 1])
    wheezing = st.selectbox("wheezing", [0, 1])

    # Auto-calculated scores
    symptom_score = sum([cough, short_breath, wheezing, anxiety])
    lifestyle_score = sum([smoking, alcohol])

    # Display auto sliders
    st.slider("SYMPTOM SCORE", 0, 10, symptom_score, key="symptom_slider", disabled=True)
    st.slider("LIFESTYLE SCORE", 0, 5, lifestyle_score, key="lifestyle_slider", disabled=True)

    if st.button("Predict Individual"):
        row = pd.DataFrame({
            "AGE": [age],
            "GENDER": [1 if gender == "Male" else 0],
            "SMOKING": [int(smoking)],
            "ANXIETY": [int(anxiety)],
            "ALCOHOL CONSUMING": [int(alcohol)],
            "COUGHING": [int(cough)],
            "SHORTNESS_OF_BREATH": [int(Short Breath)],
            "CHEST PAIN": [int(chest_pain)],
            "SYMPTOM_SCORE": [symptom_score],
            "LIFESTYLE_SCORE": [lifestyle_score]
        })
        for col in feature_names:
            if col not in row: row[col] = 0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]
        pred = int(prob > threshold_manual)
        st.success(f"Prediction: {'🛑 LUNG CANCER' if pred else '✅ NO LUNG CANCER'} (Probability: {prob:.2f})")

        # Confidence Bar Chart
        fig, ax = plt.subplots()
        bars = ax.bar(["No Lung Cancer", "Lung Cancer"], [1 - prob, prob], color=["green", "red"])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center')
        st.pyplot(fig)

        # Download CSV
        result_df = pd.DataFrame({"Prediction": ["LUNG CANCER" if pred else "NO LUNG CANCER"], "Probability": [prob]})
        st.download_button("📥 Download Result (CSV)", result_df.to_csv(index=False), "individual_result.csv", "text/csv")

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Lung Cancer Prediction Result", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}", ln=True)
        pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("📄 Download PDF", pdf_bytes, "individual_result.pdf", "application/pdf")

# ----------------------------
# ✅ PERMUTATION IMPORTANCE TOGGLE
# ----------------------------
if st.sidebar.checkbox("Show Permutation Importance"):
    try:
        result = permutation_importance(pipeline, df_input, pipeline.predict(df_input), n_repeats=5, random_state=42)
        sorted_idx = result.importances_mean.argsort()[::-1]
        fig, ax = plt.subplots()
        ax.barh(np.array(feature_names)[sorted_idx], result.importances_mean[sorted_idx])
        st.pyplot(fig)
    except:
        st.warning("Live calculation failed. Showing static chart.")
        fig, ax = plt.subplots()
        ax.barh(importance_data["Feature"], importance_data["Importance"])
        st.pyplot(fig)

# ----------------------------
# ✅ EMAIL (PLACEHOLDER)
# ----------------------------
email = st.sidebar.text_input("Enter your email")
if email and st.sidebar.button("Send Email"):
    st.sidebar.success("Email sending simulated (configure SMTP for real use)")





