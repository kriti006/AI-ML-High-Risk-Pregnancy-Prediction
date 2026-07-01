import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.facecolor'] = '#FAFAFA'
matplotlib.rcParams['axes.facecolor'] = '#FAFAFA'

st.set_page_config(
    page_title="MaternaAI — Pregnancy Risk Predictor",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F3F0;
}

.stApp {
    background: linear-gradient(160deg, #FDF6F2 0%, #F3EEF8 50%, #EEF4F2 100%);
}

/* Hide streamlit default header */
#MainMenu, footer, header { visibility: hidden; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #F9E8E8 0%, #EDE8F7 50%, #E8F2EE 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    border: 1px solid rgba(200,180,220,0.3);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #3D2C5E;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1rem;
    color: #7A6E8A;
    margin: 0;
    font-weight: 300;
}

/* Cards */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #EDE8F0;
    box-shadow: 0 2px 16px rgba(100,80,140,0.06);
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #A89DC0;
    margin-bottom: 20px;
}

/* Section divider label */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #B8A8C8;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #EDE8F0;
}

/* Result boxes */
.result-high {
    background: linear-gradient(135deg, #FDE8E8, #FAD5D5);
    border: 1px solid #F5B8B8;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
}
.result-high h2 {
    color: #C0392B;
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    margin: 0 0 8px 0;
}
.result-high p { color: #8B3030; margin: 0; font-size: 0.95rem; }

.result-safe {
    background: linear-gradient(135deg, #E8F5EE, #D5EDE0);
    border: 1px solid #A8D8BC;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
}
.result-safe h2 {
    color: #1E7A48;
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    margin: 0 0 8px 0;
}
.result-safe p { color: #2D5E40; margin: 0; font-size: 0.95rem; }

/* Probability badge */
.prob-badge {
    display: inline-block;
    background: #F3EEF8;
    border: 1px solid #D4C8E8;
    border-radius: 40px;
    padding: 6px 20px;
    font-size: 0.9rem;
    font-weight: 500;
    color: #5A4080;
    margin-top: 12px;
}

/* Input styling */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #5A4E6A !important;
    letter-spacing: 0.3px !important;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border-color: #DDD5E8 !important;
    background: #FAFAF8 !important;
}

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7C5CBF, #5A8ABF) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 16px rgba(124,92,191,0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
}

/* Warning tags */
.factor-tag {
    background: #FEF3E8;
    border-left: 3px solid #E8923A;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #7A4A1E;
}
.factor-tag strong { color: #C0622A; }

/* Footer */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #B0A8B8;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #EDE8F0;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🌸 MaternaAI</h1>
    <p>Clinical pregnancy risk assessment powered by machine learning — enter patient vitals to receive an instant risk prediction with AI-driven explanation.</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    model         = joblib.load('logistic_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, feature_names

model, feature_names = load_all()

normal_ranges = {
    'age_years':                (18, 35),
    'systolic_bp_mmHg':         (90, 120),
    'diastolic_bp_mmHg':        (60, 80),
    'random_blood_sugar_mg_dL': (70, 140),
    'body_temperature_F':       (97.0, 99.0),
    'heart_rate_bpm':           (60, 100),
    'hemoglobin_g_dL':          (11.0, 15.0),
    'hba1c_percent':            (4.0, 5.6),
    'respiratory_rate_bpm':     (12, 20),
    'bmi':                      (18.5, 24.9),
    'spo2_percent':             (95, 100),
    'edema_severity':           (0, 0),
    'symptoms_score_0_10':      (0, 3)
}

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Patient Information</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><p class="card-title">Obstetric History</p>', unsafe_allow_html=True)
    age             = st.number_input("Age (years)",             min_value=10,  max_value=60,  value=25)
    gravida         = st.number_input("Gravida (G)",             min_value=0,   max_value=15,  value=1)
    para            = st.number_input("Para (P)",                min_value=0,   max_value=15,  value=0)
    live_child      = st.number_input("Live Children (L)",       min_value=0,   max_value=15,  value=0)
    abortion        = st.number_input("Abortions (A)",           min_value=0,   max_value=10,  value=0)
    death           = st.number_input("Deaths (D)",              min_value=0,   max_value=10,  value=0)
    gestational_age = st.number_input("Gestational Age (weeks)", min_value=1,   max_value=42,  value=20)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><p class="card-title">Vital Signs</p>', unsafe_allow_html=True)
    systolic_bp        = st.number_input("Systolic BP (mmHg)",       min_value=70,   max_value=200,   value=120)
    diastolic_bp       = st.number_input("Diastolic BP (mmHg)",      min_value=40,   max_value=130,   value=80)
    heart_rate         = st.number_input("Heart Rate (bpm)",          min_value=40,   max_value=180,   value=80)
    body_temp          = st.number_input("Body Temperature (°F)",     min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    respiratory_rate   = st.number_input("Respiratory Rate (bpm)",    min_value=10,   max_value=40,    value=18)
    spo2               = st.number_input("SpO2 (%)",                  min_value=70,   max_value=100,   value=98)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><p class="card-title">Lab & Clinical</p>', unsafe_allow_html=True)
    random_blood_sugar = st.number_input("Blood Sugar (mg/dL)",  min_value=50,   max_value=400,   value=100)
    hemoglobin         = st.number_input("Hemoglobin (g/dL)",    min_value=4.0,  max_value=20.0,  value=11.0, step=0.1)
    hba1c              = st.number_input("HbA1c (%)",            min_value=3.0,  max_value=15.0,  value=5.5,  step=0.1)
    bmi                = st.number_input("BMI",                  min_value=10.0, max_value=50.0,  value=22.0, step=0.1)
    edema_severity     = st.selectbox("Edema Severity",          options=[0, 1, 2, 3], index=0)
    symptoms_score     = st.slider("Symptoms Score (0–10)",      min_value=0,    max_value=10,    value=2)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Run Risk Assessment →"):

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
        diastolic_bp > 90 or
        random_blood_sugar > 200 or
        hba1c > 7.5 or
        spo2 < 90 or
        hemoglobin < 7.0 or
        bmi > 35
    )

    prediction = 1 if (probability >= 0.30 or high_risk_rules) else 0

    st.markdown('<p class="section-label">Assessment Result</p>', unsafe_allow_html=True)

    res_col, xai_col = st.columns([1, 1.6])

    with res_col:
        if prediction == 1:
            prob_display = "Clinical threshold exceeded" if high_risk_rules else f"{round(probability * 100, 1)}% risk probability"
            st.markdown(f"""
            <div class="result-high">
                <h2>⚠ High Risk</h2>
                <p>Immediate clinical attention recommended.</p>
                <div class="prob-badge">{prob_display}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <h2>✓ Low Risk</h2>
                <p>Patient vitals appear within acceptable range.</p>
                <div class="prob-badge">{round(probability * 100, 1)}% risk probability</div>
            </div>
            """, unsafe_allow_html=True)

        # Clinical flags
        flags = []
        if systolic_bp > 140:  flags.append(f"Systolic BP {systolic_bp} mmHg — above 140 threshold")
        if diastolic_bp > 90:  flags.append(f"Diastolic BP {diastolic_bp} mmHg — above 90 threshold")
        if random_blood_sugar > 200: flags.append(f"Blood sugar {random_blood_sugar} mg/dL — above 200")
        if hba1c > 7.5:        flags.append(f"HbA1c {hba1c}% — above 7.5 threshold")
        if spo2 < 90:          flags.append(f"SpO2 {spo2}% — below 90 threshold")
        if hemoglobin < 7.0:   flags.append(f"Hemoglobin {hemoglobin} g/dL — below 7.0")
        if bmi > 35:           flags.append(f"BMI {bmi} — above 35 threshold")

        if flags:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="card-title" style="color:#C0622A;letter-spacing:1.5px;font-size:0.72rem;">CLINICAL FLAGS</p>', unsafe_allow_html=True)
            for f in flags:
                st.markdown(f'<div class="factor-tag"><strong>↑</strong> {f}</div>', unsafe_allow_html=True)

    with xai_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">🧠 Explainable AI — Why this Prediction?</p>', unsafe_allow_html=True)

        checks = [
            (systolic_bp,        90,   140,  'Systolic BP',      'mmHg'),
            (diastolic_bp,       60,   90,   'Diastolic BP',     'mmHg'),
            (random_blood_sugar, 70,   140,  'Blood Sugar',      'mg/dL'),
            (hba1c,              4.0,  5.6,  'HbA1c',            '%'),
            (hemoglobin,         11.0, 15.0, 'Hemoglobin',       'g/dL'),
            (bmi,                18.5, 24.9, 'BMI',              ''),
            (spo2,               95,   100,  'SpO2',             '%'),
            (body_temp,          97.0, 99.0, 'Body Temperature', '°F'),
            (heart_rate,         60,   100,  'Heart Rate',       'bpm'),
            (respiratory_rate,   12,   20,   'Respiratory Rate', 'bpm'),
            (symptoms_score,     0,    3,    'Symptoms Score',   ''),
            (age,                18,   35,   'Age',              'yrs'),
        ]

        high_factors  = []
        low_factors   = []
        normal_factors = []

        for val, lo, hi, label, unit in checks:
            u = f" {unit}" if unit else ""
            if val > hi:
                high_factors.append(f"<b>{label}</b> is {val}{u} — above normal range ({lo}–{hi}{u})")
            elif val < lo:
                low_factors.append(f"<b>{label}</b> is {val}{u} — below normal range ({lo}–{hi}{u})")
            else:
                normal_factors.append(f"<b>{label}</b> is {val}{u} — within normal range ✓")

        if high_factors:
            st.markdown('<p class="card-title" style="color:#C0622A;font-size:0.72rem;letter-spacing:1.5px;">⬆ ELEVATED PARAMETERS</p>', unsafe_allow_html=True)
            for f in high_factors:
                st.markdown(f'<div class="factor-tag">⬆ {f}</div>', unsafe_allow_html=True)

        if low_factors:
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<p class="card-title" style="color:#2A5EA0;font-size:0.72rem;letter-spacing:1.5px;">⬇ LOW PARAMETERS</p>', unsafe_allow_html=True)
            for f in low_factors:
                st.markdown(f'<div class="factor-tag" style="border-left-color:#5A8ABF;background:#EEF3FE;color:#1A3060;">⬇ {f}</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="card-title" style="color:#2E7D5A;font-size:0.72rem;letter-spacing:1.5px;">✓ NORMAL PARAMETERS</p>', unsafe_allow_html=True)
        for f in normal_factors:
            st.markdown(f'<div class="factor-tag" style="border-left-color:#72BFA0;background:#EEF7F2;color:#1A4A30;">✓ {f}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">MaternaAI is a clinical decision-support tool. Always confirm findings with a qualified healthcare professional.</div>', unsafe_allow_html=True)
