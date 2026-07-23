import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
from catboost import CatBoostClassifier

matplotlib.rcParams['figure.facecolor'] = '#FAFAFA'
matplotlib.rcParams['axes.facecolor'] = '#FAFAFA'

st.set_page_config(
    page_title="MaternaCare — Pregnancy Risk Predictor",
    page_icon="🌸",
    layout="wide"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@500;600&display=swap');

html,
body,
[class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F3F0;
}

.stApp {
    background: linear-gradient(
        160deg,
        #FDF6F2 0%,
        #F3EEF8 50%,
        #EEF4F2 100%
    );
}

#MainMenu,
footer,
header {
    visibility: hidden;
}


@keyframes floatSlow {
    0%   { transform: translateY(0px) translateX(0px) scale(1); }
    50%  { transform: translateY(-18px) translateX(10px) scale(1.05); }
    100% { transform: translateY(0px) translateX(0px) scale(1); }
}
@keyframes floatSlow2 {
    0%   { transform: translateY(0px) translateX(0px) scale(1); }
    50%  { transform: translateY(16px) translateX(-14px) scale(1.08); }
    100% { transform: translateY(0px) translateX(0px) scale(1); }
}
.blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(2px);
    opacity: 0.55;
    z-index: 0;
    pointer-events: none;
}
.blob-1 {
    width: 140px; height: 140px;
    background: radial-gradient(circle at 30% 30%, #F3C9C9, #F9E8E8);
    top: 10px; left: 6%;
    animation: floatSlow 7s ease-in-out infinite;
}
.blob-2 {
    width: 90px; height: 90px;
    background: radial-gradient(circle at 30% 30%, #C9B9E8, #EDE8F7);
    top: 60px; right: 10%;
    animation: floatSlow2 8s ease-in-out infinite;
}
.blob-3 {
    width: 60px; height: 60px;
    background: radial-gradient(circle at 30% 30%, #A8D8BC, #E8F2EE);
    bottom: 0px; left: 18%;
    animation: floatSlow 6s ease-in-out infinite;
}

.hero {                           
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        135deg,
        #F9E8E8 0%,
        #EDE8F7 50%,
        #E8F2EE 100%
    );
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    border: 1px solid rgba(200, 180, 220, 0.3);
}

.hero h1 {
    position: relative;
    z-index: 1;
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #3D2C5E;
    margin: 0 0 8px 0;
}

.hero p {
    position: relative;
    z-index: 1;
    font-size: 1rem;
    color: #7A6E8A;
    margin: 0;
    font-weight: 300;
}

/* Small pill badge, e.g. "AI-Powered • Doctor-reviewed ranges" */
.hero-badge {
    position: relative;
    z-index: 1;
    display: inline-block;
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(200,180,220,0.5);
    border-radius: 30px;
    padding: 6px 18px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: #5A4080;
    margin-bottom: 18px;
}

.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #EDE8F0;
    box-shadow: 0 2px 16px rgba(100, 80, 140, 0.06);
}

.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #A89DC0;
    margin-bottom: 20px;
}

.section-label {
    font-size: 0.70rem;
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
    content: "";
    flex: 1;
    height: 1px;
    background: #EDE8F0;
}


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

.result-high p {
    color: #8B3030;
    margin: 0;
    font-size: 0.95rem;
}

.result-mild {
    background: linear-gradient(135deg, #FEF9E8, #FEF0CC);
    border: 1px solid #F5D87A;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
}

.result-mild h2 {
    color: #B7770D;
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    margin: 0 0 8px 0;
}

.result-mild p {
    color: #7A5010;
    margin: 0;
    font-size: 0.95rem;
}

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

.result-safe p {
    color: #2D5E40;
    margin: 0;
    font-size: 0.95rem;
}


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


div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #5A4E6A !important;
    letter-spacing: 0.3px !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border-color: #DDD5E8 !important;
    background: #FAFAF8 !important;
    color: #333 !important;
}


div[data-testid="stButton"] > button {
    background: linear-gradient(
        135deg,
        #7C5CBF,
        #5A8ABF
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(124, 92, 191, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(124, 92, 191, 0.35) !important;
}


.factor-tag {
    background: #FEF3E8;
    border-left: 3px solid #E8923A;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #7A4A1E;
}

.factor-tag strong {
    color: #C0622A;
}


.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #B0A8B8;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #EDE8F0;
}


.legend-box {
    display: inline-block;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 0.82rem;
    font-weight: 500;
}

.debug-box {
    background: #F5F3FA;
    border: 1px dashed #C8BEDB;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 14px;
    font-size: 0.82rem;
    color: #4A4360;
}

.param-icon-btn {
    background: #FFFFFF;
    border: 1px solid #EDE8F0;
    border-radius: 20px;
    padding: 48px 32px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(100, 80, 140, 0.08);
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.param-icon-btn:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(100, 80, 140, 0.14);
}

.param-icon-btn .emoji {
    font-size: 3.4rem;
    display: block;
    margin-bottom: 16px;
}

.param-icon-btn h3 {
    font-family: 'Playfair Display', serif;
    color: #3D2C5E;
    margin: 0 0 6px 0;
}

.param-icon-btn p {
    color: #7A6E8A;
    font-size: 0.85rem;
    margin: 0;
}

/* ===== Small feature strip on login page ===== */
.feature-strip {
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 28px 0 8px 0;
}
.feature-pill {
    background: #FFFFFF;
    border: 1px solid #EDE8F0;
    border-radius: 14px;
    padding: 14px 20px;
    min-width: 150px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(100, 80, 140, 0.06);
}
.feature-pill .emoji {
    font-size: 1.6rem;
    display: block;
    margin-bottom: 6px;
}
.feature-pill .label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #5A4E6A;
}

/* ===== Mini stat cards on welcome page ===== */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #EDE8F0;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(100, 80, 140, 0.06);
}
.stat-card .num {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #3D2C5E;
    font-weight: 600;
}
.stat-card .lbl {
    font-size: 0.72rem;
    color: #A89DC0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Session State Defaults
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "started" not in st.session_state:
    st.session_state.started = False

# =========================
# STEP 1 — LOGIN PAGE
# =========================
if not st.session_state.logged_in:
    st.markdown("""
    <div class="hero" style="text-align:center; padding: 64px 48px;">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
        <div class="blob blob-3"></div>
        <div class="hero-badge">✨ AI-Powered &nbsp;•&nbsp; Doctor-reviewed ranges &nbsp;•&nbsp; 100% Private</div>
        <h1 style="font-size:3rem;">🌸 MaternaCare</h1>
        <p style="font-size:1.1rem; max-width:640px; margin:0 auto;">
            Enter your pregnancy vitals and get an instant, easy-to-understand
            risk check — right from home, in just a couple of minutes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-strip">
        <div class="feature-pill"><span class="emoji">⚡</span><span class="label">Instant Results</span></div>
        <div class="feature-pill"><span class="emoji">🩺</span><span class="label">Doctor-grade Ranges</span></div>
        <div class="feature-pill"><span class="emoji">🔒</span><span class="label">Private &amp; Secure</span></div>
        <div class="feature-pill"><span class="emoji">🧠</span><span class="label">Explainable AI</span></div>
    </div>
    """, unsafe_allow_html=True)

    l1, l2, l3 = st.columns([1, 1.2, 1])
    with l2:
        st.markdown('<div class="card"><p class="card-title">🔐 Login to Continue</p>', unsafe_allow_html=True)
        name_input = st.text_input("Your Name", placeholder="e.g. Priya Sharma")
        if st.button("Login →"):
            if name_input.strip() == "":
                st.warning("Please enter your name to continue.")
            else:
                st.session_state.user_name = name_input.strip()
                st.session_state.logged_in = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer" style="margin-top:48px;">
        MaternaCare gives a helpful early indication, not a medical diagnosis.
        Please always share these results with your doctor.
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ==============================
# STEP 2 — WELCOME PAGE WITH ICON 
# =============================
if not st.session_state.started:
    top_l, top_r = st.columns([6, 1])
    with top_r:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.started = False
            st.session_state.user_name = ""
            st.rerun()

    st.markdown(f"""
    <div class="hero" style="text-align:center; padding: 56px 48px;">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
        <div class="blob blob-3"></div>
        <div class="hero-badge">🌷 Personalized just for you</div>
        <h1 style="font-size:2.6rem;">🌸 Welcome, {st.session_state.user_name}</h1>
        <p style="font-size:1.05rem; max-width:640px; margin:0 auto;">
            Click below to enter your vitals and get an instant, easy-to-read
            check on your pregnancy risk level.
        </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown('<div class="stat-card"><div class="num">2 min</div><div class="lbl">To Complete</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat-card"><div class="num">19</div><div class="lbl">Health Signals Checked</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="stat-card"><div class="num">3</div><div class="lbl">ML Models Combined</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    i1, i2, i3 = st.columns([1, 1, 1])
    with i2:
        st.markdown("""
        <div class="param-icon-btn">
            <span class="emoji">🩺</span>
            <h3>Enter Your Health Details</h3>
            <p>Your vitals, lab reports & pregnancy history</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start My Check-up →", key="start_btn"):
            st.session_state.started = True
            st.rerun()

    st.markdown("""
    <div class="footer" style="margin-top:48px;">
        MaternaCare gives a helpful early indication, not a medical diagnosis.
        Please always share these results with your doctor.
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# =========================
# STEP 3 — PARAMETERS 
# =========================
back_l, back_r = st.columns([1, 8])
with back_l:
    if st.button("← Back"):
        st.session_state.started = False
        st.rerun()

st.markdown(f"""
<div class="hero">
<div class="blob blob-2"></div>
<h1>🌸 MaternaCare</h1>
<p>Hello {st.session_state.user_name} — enter your vitals below to get an instant risk check, with a simple explanation of why.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_all():
    model = joblib.load("voting_model_presplit_smote.pkl")
    feature_names = joblib.load("feature_names_presplit_smote.pkl")
    return model, feature_names


model, feature_names = load_all()

with st.expander("ℹ️ How is my risk level decided? (Click to see)"):

    st.markdown("""
| Parameter | Normal | Mild Risk | High Risk |
|-----------|--------|-----------|-----------|
| Systolic BP | <130 mmHg | 130–140 mmHg | >140 mmHg |
| Diastolic BP | <85 mmHg | 85–90 mmHg | >90 mmHg |
| Blood Sugar | 70–140 mg/dL | 140–200 mg/dL | >200 mg/dL |
| HbA1c | <5.7% | 5.7–6.5% | >6.5% |
| Hemoglobin | >11 g/dL | 9–11 g/dL | <9 g/dL |
| BMI | 18.5–25 | 25–30 | >30 |
| SpO₂ | >95% | 92–95% | <92% |
| Age | 18–35 years | 35–40 years | >40 years |

### Risk Interpretation

-  **High Risk:** Any parameter falls in the High Risk range or **3+** mild-risk indicators.
-  **Mild Risk:** **1–2** parameters fall in the Mild Risk range.
-  **Low Risk:** All parameters are within the Normal range.

*Boundary values (e.g. exactly 130, 140, 95) are always counted on the Mild side, never left unflagged.*
""")


st.markdown(
    '<p class="section-label">Your Information</p>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><p class="card-title">Your Pregnancy History</p>', unsafe_allow_html=True)
    age             = st.number_input("Age (years)",             min_value=10,  max_value=60,  value=25)
    gravida         = st.number_input("Gravida (G)",             min_value=0,   max_value=15,  value=1)
    para            = st.number_input("Para (P)",                min_value=0,   max_value=15,  value=0)
    live_child      = st.number_input("Live Children (L)",       min_value=0,   max_value=15,  value=0)
    abortion        = st.number_input("Abortions (A)",           min_value=0,   max_value=10,  value=0)
    death           = st.number_input("Deaths (D)",              min_value=0,   max_value=10,  value=0)
    gestational_age = st.number_input("Gestational Age (weeks)", min_value=1,   max_value=42,  value=20)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><p class="card-title">Your Vital Signs</p>', unsafe_allow_html=True)
    systolic_bp      = st.number_input("Systolic BP (mmHg)",    min_value=70,   max_value=200,   value=120)
    diastolic_bp     = st.number_input("Diastolic BP (mmHg)",   min_value=40,   max_value=130,   value=80)
    heart_rate       = st.number_input("Heart Rate (bpm)",       min_value=40,   max_value=180,   value=80)
    body_temp        = st.number_input("Body Temperature (°F)",  min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    respiratory_rate = st.number_input("Respiratory Rate (bpm)", min_value=10,   max_value=40,    value=18)
    spo2             = st.number_input("SpO2 (%)",               min_value=70,   max_value=100,   value=98)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><p class="card-title">Your Lab Reports</p>', unsafe_allow_html=True)
    random_blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50,   max_value=400,   value=100)
    hemoglobin         = st.number_input("Hemoglobin (g/dL)",   min_value=4.0,  max_value=20.0,  value=11.0, step=0.1)
    hba1c              = st.number_input("HbA1c (%)",           min_value=3.0,  max_value=15.0,  value=5.5,  step=0.1)
    bmi                = st.number_input("BMI",                 min_value=10.0, max_value=50.0,  value=22.0, step=0.1)
    edema_severity     = st.selectbox("Edema Severity",         options=[0, 1, 2, 3], index=0)
    symptoms_score     = st.slider("Symptoms Score (0–10)",     min_value=0,    max_value=10,    value=2)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Check My Risk →"):

    raw_input = {
        'age_years': age, 'gravida_G': gravida, 'para_P': para,
        'live_child_L': live_child, 'abortion_A': abortion, 'death_D': death,
        'gestational_age_weeks': gestational_age, 'systolic_bp_mmHg': systolic_bp,
        'diastolic_bp_mmHg': diastolic_bp, 'random_blood_sugar_mg_dL': random_blood_sugar,
        'body_temperature_F': body_temp, 'heart_rate_bpm': heart_rate,
        'hemoglobin_g_dL': hemoglobin, 'hba1c_percent': hba1c,
        'respiratory_rate_bpm': respiratory_rate, 'bmi': bmi,
        'spo2_percent': spo2, 'edema_severity': edema_severity,
        'symptoms_score_0_10': symptoms_score
    }

    input_df = pd.DataFrame([raw_input])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    probability = model.predict_proba(input_df)[0][1]

    # ── High Risk rules (severe abnormality) ──────────────────────────────────
    high_risk_flags = []
    if systolic_bp > 140:          high_risk_flags.append(f"Systolic BP {systolic_bp} mmHg > 140")
    if diastolic_bp > 90:          high_risk_flags.append(f"Diastolic BP {diastolic_bp} mmHg > 90")
    if random_blood_sugar > 200:   high_risk_flags.append(f"Blood Sugar {random_blood_sugar} mg/dL > 200")
    if hba1c > 6.5:                high_risk_flags.append(f"HbA1c {hba1c}% > 6.5")
    if hemoglobin < 9.0:           high_risk_flags.append(f"Hemoglobin {hemoglobin} g/dL < 9.0")
    if bmi > 30:                   high_risk_flags.append(f"BMI {bmi} > 30")
    if spo2 < 92:                  high_risk_flags.append(f"SpO2 {spo2}% < 92")
    if age > 40:                   high_risk_flags.append(f"Age {age} yrs > 40")
    if edema_severity >= 3:        high_risk_flags.append(f"Edema Severity {edema_severity} (severe)")

    # ── Mild Risk rules (moderate abnormality) — inclusive boundaries ─────────
    mild_risk_flags = []
    if 130 <= systolic_bp <= 140:          mild_risk_flags.append(f"Systolic BP {systolic_bp} mmHg (130–140)")
    if 85 <= diastolic_bp <= 90:           mild_risk_flags.append(f"Diastolic BP {diastolic_bp} mmHg (85–90)")
    if 140 < random_blood_sugar <= 200:    mild_risk_flags.append(f"Blood Sugar {random_blood_sugar} mg/dL (140–200)")
    if 5.7 <= hba1c <= 6.5:                mild_risk_flags.append(f"HbA1c {hba1c}% (5.7–6.5)")
    if 9.0 <= hemoglobin <= 11.0:          mild_risk_flags.append(f"Hemoglobin {hemoglobin} g/dL (9–11)")
    if 25 < bmi <= 30:                     mild_risk_flags.append(f"BMI {bmi} (25–30)")
    if 92 <= spo2 <= 95:                   mild_risk_flags.append(f"SpO2 {spo2}% (92–95)")
    if 35 < age <= 40:                     mild_risk_flags.append(f"Age {age} yrs (35–40)")
    if edema_severity == 2:                mild_risk_flags.append(f"Edema Severity {edema_severity} (moderate)")
    if symptoms_score >= 6:                mild_risk_flags.append(f"Symptoms Score {symptoms_score}/10")

    # ── Determine risk level ──────────────────────────────────────────────────
    if high_risk_flags or probability >= 0.30 or len(mild_risk_flags) >= 3:
        risk_level = "HIGH"
    elif mild_risk_flags:
        risk_level = "MILD"
    else:
        risk_level = "LOW"

    st.markdown('<p class="section-label">Assessment Result</p>', unsafe_allow_html=True)

    res_col, xai_col = st.columns([1, 1.6])

    with res_col:
        if risk_level == "HIGH":
            st.markdown(f"""
            <div class="result-high">
                <h2>⚠ High Risk</h2>
                <p>Please contact your doctor as soon as possible.</p>
                <div class="prob-badge">ML Probability: {round(probability * 100, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)
            if high_risk_flags:
                st.markdown('<p class="card-title" style="color:#C0392B;letter-spacing:1.5px;font-size:0.72rem;margin-top:16px;">🔴 HIGH RISK FLAGS</p>', unsafe_allow_html=True)
                for f in high_risk_flags:
                    st.markdown(f'<div class="factor-tag"><strong>↑</strong> {f}</div>', unsafe_allow_html=True)
            if mild_risk_flags:
                st.markdown('<p class="card-title" style="color:#B7770D;letter-spacing:1.5px;font-size:0.72rem;margin-top:16px;">🟡 ALSO MILD</p>', unsafe_allow_html=True)
                for f in mild_risk_flags:
                    st.markdown(f'<div class="factor-tag" style="border-left-color:#E8C43A;background:#FEF9E8;color:#7A5010;"><strong>!</strong> {f}</div>', unsafe_allow_html=True)

        elif risk_level == "MILD":
            st.markdown(f"""
            <div class="result-mild">
                <h2>⚡ Mild Risk</h2>
                <p>A few things need watching. Please plan a follow-up with your doctor.</p>
                <div class="prob-badge">ML Probability: {round(probability * 100, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)
            if mild_risk_flags:
                st.markdown('<p class="card-title" style="color:#B7770D;letter-spacing:1.5px;font-size:0.72rem;margin-top:16px;">🟡 MILD RISK FLAGS</p>', unsafe_allow_html=True)
                for f in mild_risk_flags:
                    st.markdown(f'<div class="factor-tag" style="border-left-color:#E8C43A;background:#FEF9E8;color:#7A5010;"><strong>!</strong> {f}</div>', unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-safe">
                <h2>✓ Low Risk</h2>
                <p>Your vitals look within the normal range. Keep up your routine check-ups!</p>
                <div class="prob-badge">ML Probability: {round(probability * 100, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="debug-box">
            <b>Quick summary</b><br>
            Serious concerns found: {len(high_risk_flags)}<br>
            Minor concerns found: {len(mild_risk_flags)}<br>
            AI confidence in risk: {round(probability * 100, 1)}%
        </div>
        """, unsafe_allow_html=True)

    with xai_col:
        st.markdown('<div class="card"><p class="card-title">🧠 Why This Result? (In Simple Terms)</p>', unsafe_allow_html=True)

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

        high_factors   = []
        low_factors    = []
        normal_factors = []

        for val, lo, hi, label, unit in checks:
            u = f" {unit}" if unit else ""
            if val > hi:
                high_factors.append(f"<b>{label}</b> is {val}{u} — above normal ({lo}–{hi}{u})")
            elif val < lo:
                low_factors.append(f"<b>{label}</b> is {val}{u} — below normal ({lo}–{hi}{u})")
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

st.markdown('<div class="footer">MaternaCare gives a helpful early indication, not a medical diagnosis. Please always share these results with your doctor.</div>', unsafe_allow_html=True)