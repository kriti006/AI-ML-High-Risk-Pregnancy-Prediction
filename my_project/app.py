import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="High Risk Pregnancy Predictor",
    page_icon="🤰",
    layout="centered"
)

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .title {
            text-align: center;
            color: #c0392b;
            font-size: 2.2rem;
            font-weight: bold;
            margin-bottom: 0px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 1rem;
            margin-bottom: 30px;
        }
        .section-header {
            background-color: #c0392b;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🤰 High Risk Pregnancy Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient health details to predict pregnancy risk using AI</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_all():
    model         = joblib.load('logistic_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, feature_names

model, feature_names = load_all()

st.markdown('<div class="section-header">📋 Patient Health Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age                = st.number_input("🎂 Age (years)",               min_value=10,   max_value=60,    value=25)
    gravida            = st.number_input("🤱 Gravida (G)",               min_value=0,    max_value=15,    value=1)
    para               = st.number_input("👶 Para (P)",                  min_value=0,    max_value=15,    value=0)
    live_child         = st.number_input("🧒 Live Children (L)",         min_value=0,    max_value=15,    value=0)
    abortion           = st.number_input("⚠️ Abortions (A)",             min_value=0,    max_value=10,    value=0)
    death              = st.number_input("💔 Deaths (D)",                min_value=0,    max_value=10,    value=0)
    gestational_age    = st.number_input("📅 Gestational Age (weeks)",   min_value=1,    max_value=42,    value=20)
    systolic_bp        = st.number_input("🩺 Systolic BP (mmHg)",        min_value=70,   max_value=200,   value=120)
    diastolic_bp       = st.number_input("🩺 Diastolic BP (mmHg)",       min_value=40,   max_value=130,   value=80)
    random_blood_sugar = st.number_input("🩸 Blood Sugar (mg/dL)",       min_value=50,   max_value=400,   value=100)

with col2:
    body_temp          = st.number_input("🌡️ Body Temperature (F)",      min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    heart_rate         = st.number_input("❤️ Heart Rate (bpm)",          min_value=40,   max_value=180,   value=80)
    hemoglobin         = st.number_input("💉 Hemoglobin (g/dL)",         min_value=4.0,  max_value=20.0,  value=11.0, step=0.1)
    hba1c              = st.number_input("🔬 HbA1c (%)",                 min_value=3.0,  max_value=15.0,  value=5.5,  step=0.1)
    respiratory_rate   = st.number_input("🫁 Respiratory Rate (bpm)",    min_value=10,   max_value=40,    value=18)
    bmi                = st.number_input("⚖️ BMI",                       min_value=10.0, max_value=50.0,  value=22.0, step=0.1)
    spo2               = st.number_input("🫀 SpO2 (%)",                  min_value=70,   max_value=100,   value=98)
    edema_severity     = st.selectbox("🦵 Edema Severity",               options=[0, 1, 2, 3], index=0)
    symptoms_score     = st.slider("📊 Symptoms Score (0-10)",           min_value=0,    max_value=10,    value=2)

st.markdown("---")

if st.button("🔍 Predict Risk", use_container_width=True):

    raw_input = {
        'age_years':                age,
        'gravida_G':                gravida,
        'para_P':                   para,
        'live_child_L':             live_child,
        'abortion_A':               abortion,
        'death_D':                  death,
        'gestational_age_weeks':    gestational_age,
        'systolic_bp_mmHg':         systolic_bp,
        'diastolic_bp_mmHg':        diastolic_bp,
        'random_blood_sugar_mg_dL': random_blood_sugar,
        'body_temperature_F':       body_temp,
        'heart_rate_bpm':           heart_rate,
        'hemoglobin_g_dL':          hemoglobin,
        'hba1c_percent':            hba1c,
        'respiratory_rate_bpm':     respiratory_rate,
        'bmi':                      bmi,
        'spo2_percent':             spo2,
        'edema_severity':           edema_severity,
        'symptoms_score_0_10':      symptoms_score
    }

    input_df = pd.DataFrame([raw_input])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    probability = model.predict_proba(input_df)[0][1]

    high_risk_rules = (
        systolic_bp > 140 or
        diastolic_bp > 90  or
        random_blood_sugar > 200 or
        hba1c > 7.5 or
        spo2 < 90 or
        hemoglobin < 7.0 or
        bmi > 35
    )

    prediction = 1 if (probability >= 0.30 or high_risk_rules) else 0

    st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.error("🔴  HIGH RISK DETECTED")
        if high_risk_rules:
            st.metric(label="Risk Probability", value="High ⚠️", help="Flagged by clinical rules")
        else:
            st.metric(label="Risk Probability", value=f"{round(probability * 100, 2)}%")
        st.warning("⚠️ This patient requires immediate medical attention and close monitoring.")
    else:
        st.success("🟢  NO RISK DETECTED")
        st.metric(label="Risk Probability", value=f"{round(probability * 100, 2)}%")
        st.info("✅ Patient appears to be in a normal condition. Routine checkup recommended.")

    st.markdown("---")
    st.markdown('<div class="section-header">🧠 Why this Prediction? (Explainable AI)</div>', unsafe_allow_html=True)
    st.markdown("The chart below shows which health factors influenced the prediction the most.")

    coefficients = model.coef_[0]
    input_values = input_df.values[0]
    impact       = coefficients * input_values

    xai_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact':  impact
    })
    xai_df = xai_df[xai_df['Impact'] != 0]
    xai_df = xai_df.sort_values('Impact', key=abs, ascending=False).head(10)

    if xai_df.empty:
        st.warning("Not enough data to show explanation.")
    else:
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in xai_df['Impact']]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(xai_df['Feature'], xai_df['Impact'], color=colors, edgecolor='white', height=0.6)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Impact on Prediction', fontsize=11)
        ax.set_title('Top Factors Influencing This Prediction', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.error("🔴 Red bars → Push towards HIGH RISK")
        with col_b:
            st.success("🟢 Green bars → Push towards NO RISK")

        st.caption("Longer bar = More impact on the prediction")

        top_feature = xai_df.iloc[0]['Feature']
        top_value   = xai_df.iloc[0]['Impact']
        direction   = "increased" if top_value > 0 else "decreased"
        st.info(f"📌 Most influential factor: **{top_feature}** — it {direction} the risk for this patient.")

    st.markdown("---")
    st.caption("This tool is for medical assistance only. Always consult a qualified doctor for diagnosis.")