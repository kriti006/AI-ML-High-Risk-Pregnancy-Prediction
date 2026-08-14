import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import hashlib
import os
import datetime
from catboost import CatBoostClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "voting_model_presplit_smote.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names_presplit_smote.pkl"))

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

div[data-testid="stSelectbox"] {
    max-width: 160px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Database Setup
# =========================
DB_PATH = "maternacare.db"

# Usernames listed here get an "Admin Dashboard" option after logging in,
# from which they can see every registered user and every check-up ever
# saved (not just their own). Add your own username to this set.
ADMIN_USERNAMES = {"admin","kriti_001"}

RAW_INPUT_COLUMNS = [
    'age_years', 'gravida_G', 'para_P', 'live_child_L', 'abortion_A', 'death_D',
    'gestational_age_weeks', 'systolic_bp_mmHg', 'diastolic_bp_mmHg',
    'random_blood_sugar_mg_dL', 'body_temperature_F', 'heart_rate_bpm',
    'hemoglobin_g_dL', 'hba1c_percent', 'respiratory_rate_bpm', 'bmi',
    'spo2_percent', 'edema_severity', 'symptoms_score_0_10'
]


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        full_name TEXT,
        salt TEXT,
        password_hash TEXT,
        created_at TEXT
    )""")
    cols_sql = ", ".join([f'"{col}" REAL' for col in RAW_INPUT_COLUMNS])
    c.execute(f"""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        created_at TEXT,
        {cols_sql},
        risk_level TEXT,
        probability REAL
    )""")
    conn.commit()
    conn.close()


def hash_password(password, salt_hex=None):
    if salt_hex is None:
        salt_hex = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256', password.encode('utf-8'), bytes.fromhex(salt_hex), 100_000
    ).hex()
    return salt_hex, pwd_hash


def verify_password(password, salt_hex, pwd_hash):
    _, check_hash = hash_password(password, salt_hex)
    return check_hash == pwd_hash


def username_exists(username):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row is not None


def create_user(username, full_name, password):
    salt_hex, pwd_hash = hash_password(password)
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (username, full_name, salt, password_hash, created_at) VALUES (?,?,?,?,?)",
        (username, full_name, salt_hex, pwd_hash, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def authenticate(username, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT full_name, salt, password_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    full_name, salt_hex, pwd_hash = row
    if verify_password(password, salt_hex, pwd_hash):
        return full_name
    return None


def save_prediction(username, raw_input, risk_level, probability):
    conn = get_conn()
    c = conn.cursor()
    cols = ["username", "created_at"] + RAW_INPUT_COLUMNS + ["risk_level", "probability"]
    placeholders = ",".join(["?"] * len(cols))
    colnames = ",".join([f'"{col}"' for col in cols])
    values = [username, datetime.datetime.now().isoformat()]
    values += [raw_input[col] for col in RAW_INPUT_COLUMNS]
    values += [risk_level, probability]
    c.execute(f"INSERT INTO predictions ({colnames}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


def get_history(username, limit=15):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE username=? ORDER BY created_at DESC LIMIT ?",
        conn, params=(username, limit)
    )
    conn.close()
    if not df.empty:
        df = df.iloc[::-1].reset_index(drop=True)
    return df


init_db()

# =========================
# Static Text (English)
# =========================
APP_TITLE = "🌸 MaternaCare"
TAGLINE = "Enter your pregnancy vitals and get an instant, easy-to-understand risk check — right from home, in just a couple of minutes."
BADGE = "✨ AI-Powered • Doctor-reviewed ranges • 100% Private"
FEAT_INSTANT = "Instant Results"
FEAT_DOCTOR = "Doctor-grade Ranges"
FEAT_PRIVATE = "Private & Secure"
FEAT_EXPLAIN = "Explainable AI"
LOGIN_TAB = "Login"
SIGNUP_TAB = "Sign Up"
USERNAME_LABEL = "Username"
PASSWORD_LABEL = "Password"
FULL_NAME_LABEL = "Your Full Name"
LOGIN_BTN = "Login →"
SIGNUP_BTN = "Create Account →"
LOGIN_TITLE = "🔐 Login to Continue"
SIGNUP_TITLE = "📝 Create a New Account"
LOGIN_ERROR = "Invalid username or password."
SIGNUP_USERNAME_TAKEN = "This username is already taken. Please choose another."
SIGNUP_MISSING = "Please fill in all fields."
SIGNUP_SUCCESS = "Account created! Please log in from the Login tab."
WELCOME_SUB = "Click below to enter your vitals and get an instant, easy-to-read check on your pregnancy risk level."
PERSONALIZED = "🌷 Personalized just for you"
STAT_TIME = "2 min"
STAT_TIME_LBL = "To Complete"
STAT_SIGNALS = "19"
STAT_SIGNALS_LBL = "Health Signals Checked"
STAT_MODELS = "3"
STAT_MODELS_LBL = "ML Models Combined"
START_BTN = "Start My Check-up →"
HISTORY_BTN = "📊 View My History"
ADMIN_BTN = "🛡️ Admin Dashboard"
START_CARD_TITLE = "Enter Your Health Details"
START_CARD_SUB = "Your vitals, lab reports & pregnancy history"
LOGOUT = "Logout"
BACK = "← Back"
FOOTER = "MaternaCare gives a helpful early indication, not a medical diagnosis. Please always share these results with your doctor."
HOW_DECIDED = "ℹ️ How is my risk level decided? (Click to see)"
RISK_TABLE_MD = """
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
"""
SECTION_YOUR_INFO = "Your Information"
CARD_HISTORY = "Your Pregnancy History"
CARD_VITALS = "Your Vital Signs"
CARD_LABS = "Your Lab Reports"
CHECK_RISK_BTN = "Check My Risk →"
SECTION_RESULT = "Assessment Result"
RESULT_HIGH_TITLE = "⚠ High Risk"
RESULT_HIGH_MSG = "Please contact your doctor as soon as possible."
RESULT_MILD_TITLE = "⚡ Mild Risk"
RESULT_MILD_MSG = "A few things need watching. Please plan a follow-up with your doctor."
RESULT_LOW_TITLE = "✓ Low Risk"
RESULT_LOW_MSG = "Your vitals look within the normal range. Keep up your routine check-ups!"
ML_PROB = "ML Probability"
HIGH_FLAGS_TITLE = "🔴 HIGH RISK FLAGS"
MILD_ALSO_TITLE = "🟡 ALSO MILD"
MILD_FLAGS_TITLE = "🟡 MILD RISK FLAGS"
QUICK_SUMMARY = "Quick summary"
SERIOUS_CONCERNS = "Serious concerns found"
MINOR_CONCERNS = "Minor concerns found"
AI_CONFIDENCE = "AI confidence in risk"
SAVED_NOTE = "✓ Saved to your history"
WHY_RESULT = "🧠 Why This Result? (In Simple Terms)"
ELEVATED_PARAMS = "⬆ ELEVATED PARAMETERS"
LOW_PARAMS = "⬇ LOW PARAMETERS"
NORMAL_PARAMS = "✓ NORMAL PARAMETERS"
HISTORY_TITLE = "📊 Your Vital Progress"
HISTORY_SUB = "Track how your key health indicators have changed across your recent check-ups."
NO_HISTORY = "You don't have any saved check-ups yet. Complete a risk check to start building your history."
HISTORY_TABLE_TITLE = "📋 Your Recent Check-ups"
BACK_TO_HOME = "← Back to Home"
ABOVE_NORMAL = "above normal"
BELOW_NORMAL = "below normal"
WITHIN_NORMAL = "within normal range ✓"
SEVERE = "severe"
MODERATE = "moderate"
YRS = "yrs"
COL_DATE = "Date"
COL_RISK = "Risk Level"
COL_PROB = "Probability (%)"

ADMIN_TITLE = "🛡️ Admin Dashboard"
ADMIN_SUB = "Every registered user and every saved check-up, all in one place."
ADMIN_USERS_TITLE = "👥 Registered Users"
ADMIN_ALL_CHECKUPS_TITLE = "📋 All Check-ups (Every User)"
ADMIN_SELECT_USER = "View check-ups for a specific user"
ADMIN_ALL_OPTION = "— All users —"
ADMIN_NO_USERS = "No users have registered yet."
ADMIN_NO_CHECKUPS = "No check-ups have been recorded yet."
COL_USERNAME = "Username"
COL_FULL_NAME = "Full Name"
COL_JOINED = "Joined"
COL_CHECKUPS = "Check-ups Done"


def welcome_msg(name):
    return f"Welcome, {name}"


FIELD_LABELS = {
    "age": "Age (years)",
    "gravida": "Gravida (G)",
    "para": "Para (P)",
    "live_child": "Live Children (L)",
    "abortion": "Abortions (A)",
    "death": "Deaths (D)",
    "gestational_age": "Gestational Age (weeks)",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "heart_rate": "Heart Rate (bpm)",
    "body_temp": "Body Temperature (°F)",
    "respiratory_rate": "Respiratory Rate (bpm)",
    "spo2": "SpO2 (%)",
    "random_blood_sugar": "Blood Sugar (mg/dL)",
    "hemoglobin": "Hemoglobin (g/dL)",
    "hba1c": "HbA1c (%)",
    "bmi": "BMI",
    "edema_severity": "Edema Severity",
    "symptoms_score": "Symptoms Score (0–10)",
}


def fl(key):
    return FIELD_LABELS[key]


SHORT_FIELD_LABELS = {
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "random_blood_sugar": "Blood Sugar",
    "hba1c": "HbA1c",
    "hemoglobin": "Hemoglobin",
    "bmi": "BMI",
    "spo2": "SpO2",
    "body_temp": "Body Temperature",
    "heart_rate": "Heart Rate",
    "respiratory_rate": "Respiratory Rate",
    "symptoms_score": "Symptoms Score",
    "age": "Age",
    "edema_severity": "Edema Severity",
}


def sfl(key):
    return SHORT_FIELD_LABELS[key]


HIST_TABLE_COLS = ["created_at"] + RAW_INPUT_COLUMNS + ["risk_level", "probability"]

# Maps each raw DB column to its FIELD_LABELS key, so the history table can
# show every recorded vital with a friendly header.
RAW_COL_TO_FL_KEY = {
    "age_years": "age",
    "gravida_G": "gravida",
    "para_P": "para",
    "live_child_L": "live_child",
    "abortion_A": "abortion",
    "death_D": "death",
    "gestational_age_weeks": "gestational_age",
    "systolic_bp_mmHg": "systolic_bp",
    "diastolic_bp_mmHg": "diastolic_bp",
    "random_blood_sugar_mg_dL": "random_blood_sugar",
    "body_temperature_F": "body_temp",
    "heart_rate_bpm": "heart_rate",
    "hemoglobin_g_dL": "hemoglobin",
    "hba1c_percent": "hba1c",
    "respiratory_rate_bpm": "respiratory_rate",
    "bmi": "bmi",
    "spo2_percent": "spo2",
    "edema_severity": "edema_severity",
    "symptoms_score_0_10": "symptoms_score",
}


def do_logout():
    st.session_state.logged_in = False
    st.session_state.started = False
    st.session_state.username = ""
    st.session_state.user_name = ""
    st.session_state.view = "welcome"


def is_admin(username):
    return username in ADMIN_USERNAMES


def get_all_users():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT username, full_name, created_at FROM users ORDER BY created_at DESC",
        conn
    )
    conn.close()
    return df


def get_all_predictions(limit=500):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_checkup_counts():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT username, COUNT(*) as checkup_count FROM predictions GROUP BY username",
        conn
    )
    conn.close()
    return dict(zip(df["username"], df["checkup_count"])) if not df.empty else {}


def format_predictions_table(df):
    table_df = df[["username"] + HIST_TABLE_COLS].copy()
    table_df["created_at"] = pd.to_datetime(table_df["created_at"]).dt.strftime('%d %b %Y, %I:%M %p')
    table_df["probability"] = (table_df["probability"].astype(float) * 100).round(1)

    col_rename = {
        "username": COL_USERNAME,
        "created_at": COL_DATE,
        "risk_level": COL_RISK,
        "probability": COL_PROB,
    }
    for raw_col, fl_key in RAW_COL_TO_FL_KEY.items():
        col_rename[raw_col] = fl(fl_key)

    table_df = table_df.rename(columns=col_rename)
    return table_df


# =========================
# Session State Defaults
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "started" not in st.session_state:
    st.session_state.started = False
if "view" not in st.session_state:
    st.session_state.view = "welcome"

# =========================
# STEP 1 — LOGIN / SIGN UP PAGE
# =========================
if not st.session_state.logged_in:

    st.markdown(f"""
    <div class="hero" style="text-align:center; padding: 64px 48px;">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
        <div class="blob blob-3"></div>
        <div class="hero-badge">{BADGE}</div>
        <h1 style="font-size:3rem;">{APP_TITLE}</h1>
        <p style="font-size:1.1rem; max-width:640px; margin:0 auto;">
            {TAGLINE}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="feature-strip">
        <div class="feature-pill"><span class="emoji">⚡</span><span class="label">{FEAT_INSTANT}</span></div>
        <div class="feature-pill"><span class="emoji">🩺</span><span class="label">{FEAT_DOCTOR}</span></div>
        <div class="feature-pill"><span class="emoji">🔒</span><span class="label">{FEAT_PRIVATE}</span></div>
        <div class="feature-pill"><span class="emoji">🧠</span><span class="label">{FEAT_EXPLAIN}</span></div>
    </div>
    """, unsafe_allow_html=True)

    l1, l2, l3 = st.columns([1, 1.2, 1])
    with l2:
        tab_login, tab_signup = st.tabs([LOGIN_TAB, SIGNUP_TAB])

        with tab_login:
            st.markdown(f'<div class="card"><p class="card-title">{LOGIN_TITLE}</p>', unsafe_allow_html=True)
            login_username = st.text_input(USERNAME_LABEL, key="login_username")
            login_password = st.text_input(PASSWORD_LABEL, type="password", key="login_password")
            if st.button(LOGIN_BTN, key="login_button"):
                if login_username.strip() == "" or login_password == "":
                    st.warning(SIGNUP_MISSING)
                else:
                    full_name = authenticate(login_username.strip(), login_password)
                    if full_name:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username.strip()
                        st.session_state.user_name = full_name
                        st.session_state.view = "welcome"
                        st.session_state.started = False
                        st.rerun()
                    else:
                        st.error(LOGIN_ERROR)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_signup:
            st.markdown(f'<div class="card"><p class="card-title">{SIGNUP_TITLE}</p>', unsafe_allow_html=True)
            signup_name = st.text_input(FULL_NAME_LABEL, key="signup_name")
            signup_username = st.text_input(USERNAME_LABEL, key="signup_username")
            signup_password = st.text_input(PASSWORD_LABEL, type="password", key="signup_password")
            if st.button(SIGNUP_BTN, key="signup_button"):
                if not signup_name.strip() or not signup_username.strip() or not signup_password:
                    st.warning(SIGNUP_MISSING)
                elif username_exists(signup_username.strip()):
                    st.error(SIGNUP_USERNAME_TAKEN)
                else:
                    create_user(signup_username.strip(), signup_name.strip(), signup_password)
                    st.success(SIGNUP_SUCCESS)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer" style="margin-top:48px;">
        {FOOTER}
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ==============================
# STEP 2 — WELCOME PAGE
# ==============================
if st.session_state.view == "welcome" and not st.session_state.started:

    top_l, top_r = st.columns([6, 1])
    with top_r:
        if st.button(LOGOUT, key="logout_btn_welcome"):
            do_logout()
            st.rerun()

    st.markdown(f"""
    <div class="hero" style="text-align:center; padding: 56px 48px;">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
        <div class="blob blob-3"></div>
        <div class="hero-badge">{PERSONALIZED}</div>
        <h1 style="font-size:2.6rem;">{welcome_msg(st.session_state.user_name)}</h1>
        <p style="font-size:1.05rem; max-width:640px; margin:0 auto;">
            {WELCOME_SUB}
        </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="num">{STAT_TIME}</div><div class="lbl">{STAT_TIME_LBL}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="num">{STAT_SIGNALS}</div><div class="lbl">{STAT_SIGNALS_LBL}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="num">{STAT_MODELS}</div><div class="lbl">{STAT_MODELS_LBL}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    i1, i2, i3 = st.columns([1, 1, 1])
    with i2:
        st.markdown(f"""
        <div class="param-icon-btn">
            <span class="emoji">🩺</span>
            <h3>{START_CARD_TITLE}</h3>
            <p>{START_CARD_SUB}</p>
        </div>
        """, unsafe_allow_html=True)

        if is_admin(st.session_state.username):
            bcol1, bcol2, bcol3 = st.columns(3)
        else:
            bcol1, bcol2 = st.columns(2)
            bcol3 = None
        with bcol1:
            if st.button(START_BTN, key="start_btn"):
                st.session_state.started = True
                st.rerun()
        with bcol2:
            if st.button(HISTORY_BTN, key="history_btn_welcome"):
                st.session_state.view = "history"
                st.rerun()
        if bcol3 is not None:
            with bcol3:
                if st.button(ADMIN_BTN, key="admin_btn_welcome"):
                    st.session_state.view = "admin"
                    st.rerun()

    st.markdown(f"""
    <div class="footer" style="margin-top:48px;">
        {FOOTER}
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ==============================
# STEP 2b — HISTORY PAGE
# ==============================
if st.session_state.view == "history":

    top_l, top_r = st.columns([6, 1])
    with top_r:
        if st.button(LOGOUT, key="logout_btn_history"):
            do_logout()
            st.rerun()

    if st.button(BACK_TO_HOME, key="back_home_from_history"):
        st.session_state.view = "welcome"
        st.rerun()

    st.markdown(f"""
    <div class="hero">
    <div class="blob blob-2"></div>
    <h1>{HISTORY_TITLE}</h1>
    <p>{HISTORY_SUB}</p>
    </div>
    """, unsafe_allow_html=True)

    hist_df = get_history(st.session_state.username, limit=15)

    if hist_df.empty:
        st.info(NO_HISTORY)
    else:
        st.markdown(f'<p class="section-label">{HISTORY_TABLE_TITLE}</p>', unsafe_allow_html=True)

        table_df = hist_df[HIST_TABLE_COLS].copy()
        table_df['created_at'] = pd.to_datetime(table_df['created_at']).dt.strftime('%d %b %Y, %I:%M %p')
        table_df['probability'] = (table_df['probability'].astype(float) * 100).round(1)

        col_rename = {"created_at": COL_DATE, "risk_level": COL_RISK, "probability": COL_PROB}
        for raw_col, fl_key in RAW_COL_TO_FL_KEY.items():
            col_rename[raw_col] = fl(fl_key)

        table_df = table_df.rename(columns=col_rename)
        table_df = table_df.iloc[::-1].reset_index(drop=True)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown(f'<div class="footer">{FOOTER}</div>', unsafe_allow_html=True)
    st.stop()

# ==============================
# STEP 2c — ADMIN DASHBOARD (owner only)
# ==============================
if st.session_state.view == "admin":

    # Guard: only usernames in ADMIN_USERNAMES may view this page, even if
    # someone tries to reach it by manipulating session state.
    if not is_admin(st.session_state.username):
        st.session_state.view = "welcome"
        st.rerun()

    top_l, top_r = st.columns([6, 1])
    with top_r:
        if st.button(LOGOUT, key="logout_btn_admin"):
            do_logout()
            st.rerun()

    if st.button(BACK_TO_HOME, key="back_home_from_admin"):
        st.session_state.view = "welcome"
        st.rerun()

    st.markdown(f"""
    <div class="hero">
    <div class="blob blob-2"></div>
    <h1>{ADMIN_TITLE}</h1>
    <p>{ADMIN_SUB}</p>
    </div>
    """, unsafe_allow_html=True)

    users_df = get_all_users()
    all_preds_df = get_all_predictions()
    checkup_counts = get_checkup_counts()

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f'<div class="stat-card"><div class="num">{len(users_df)}</div><div class="lbl">Total Users</div></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div class="stat-card"><div class="num">{len(all_preds_df)}</div><div class="lbl">Total Check-ups</div></div>', unsafe_allow_html=True)
    with a3:
        high_count = int((all_preds_df["risk_level"] == "HIGH").sum()) if not all_preds_df.empty else 0
        st.markdown(f'<div class="stat-card"><div class="num">{high_count}</div><div class="lbl">High Risk Check-ups</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-label">{ADMIN_USERS_TITLE}</p>', unsafe_allow_html=True)

    if users_df.empty:
        st.info(ADMIN_NO_USERS)
    else:
        users_display = users_df.copy()
        users_display["created_at"] = pd.to_datetime(users_display["created_at"]).dt.strftime('%d %b %Y, %I:%M %p')
        users_display["checkup_count"] = users_display["username"].map(checkup_counts).fillna(0).astype(int)
        users_display = users_display.rename(columns={
            "username": COL_USERNAME,
            "full_name": COL_FULL_NAME,
            "created_at": COL_JOINED,
            "checkup_count": COL_CHECKUPS,
        })
        st.dataframe(users_display, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-label">{ADMIN_ALL_CHECKUPS_TITLE}</p>', unsafe_allow_html=True)

    if all_preds_df.empty:
        st.info(ADMIN_NO_CHECKUPS)
    else:
        user_options = [ADMIN_ALL_OPTION] + sorted(all_preds_df["username"].unique().tolist())
        selected_user = st.selectbox(ADMIN_SELECT_USER, options=user_options, key="admin_user_filter")

        filtered_df = all_preds_df if selected_user == ADMIN_ALL_OPTION else all_preds_df[all_preds_df["username"] == selected_user]

        table_df = format_predictions_table(filtered_df)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown(f'<div class="footer">{FOOTER}</div>', unsafe_allow_html=True)
    st.stop()

# =========================
# STEP 3 — PARAMETERS
# =========================

back_l, back_r, logout_r = st.columns([1, 6, 1])
with back_l:
    if st.button(BACK, key="back_btn"):
        st.session_state.started = False
        st.rerun()
with logout_r:
    if st.button(LOGOUT, key="logout_btn_form"):
        do_logout()
        st.rerun()

st.markdown(f"""
<div class="hero">
<div class="blob blob-2"></div>
<h1>{APP_TITLE}</h1>
<p>{welcome_msg(st.session_state.user_name)} — {WELCOME_SUB}</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_all():
    model = joblib.load(os.path.join(BASE_DIR, "voting_model_presplit_smote.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names_presplit_smote.pkl"))
    return model, feature_names


model, feature_names = load_all()

with st.expander(HOW_DECIDED):
    st.markdown(RISK_TABLE_MD)

st.markdown(
    f'<p class="section-label">{SECTION_YOUR_INFO}</p>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="card"><p class="card-title">{CARD_HISTORY}</p>', unsafe_allow_html=True)
    age             = st.number_input(fl("age"),             min_value=10,  max_value=60,  value=25)
    gravida         = st.number_input(fl("gravida"),         min_value=0,   max_value=15,  value=1)
    para            = st.number_input(fl("para"),            min_value=0,   max_value=15,  value=0)
    live_child      = st.number_input(fl("live_child"),      min_value=0,   max_value=15,  value=0)
    abortion        = st.number_input(fl("abortion"),        min_value=0,   max_value=10,  value=0)
    death           = st.number_input(fl("death"),           min_value=0,   max_value=10,  value=0)
    gestational_age = st.number_input(fl("gestational_age"), min_value=1,   max_value=42,  value=20)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="card"><p class="card-title">{CARD_VITALS}</p>', unsafe_allow_html=True)
    systolic_bp      = st.number_input(fl("systolic_bp"),      min_value=70,   max_value=200,   value=120)
    diastolic_bp     = st.number_input(fl("diastolic_bp"),     min_value=40,   max_value=130,   value=80)
    heart_rate       = st.number_input(fl("heart_rate"),       min_value=40,   max_value=180,   value=80)
    body_temp        = st.number_input(fl("body_temp"),        min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    respiratory_rate = st.number_input(fl("respiratory_rate"), min_value=10,   max_value=40,    value=18)
    spo2             = st.number_input(fl("spo2"),             min_value=70,   max_value=100,   value=98)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="card"><p class="card-title">{CARD_LABS}</p>', unsafe_allow_html=True)
    random_blood_sugar = st.number_input(fl("random_blood_sugar"), min_value=50,   max_value=400,   value=100)
    hemoglobin         = st.number_input(fl("hemoglobin"),         min_value=4.0,  max_value=20.0,  value=11.0, step=0.1)
    hba1c              = st.number_input(fl("hba1c"),              min_value=3.0,  max_value=15.0,  value=5.5,  step=0.1)
    bmi                = st.number_input(fl("bmi"),                min_value=10.0, max_value=50.0,  value=22.0, step=0.1)
    edema_severity     = st.selectbox(fl("edema_severity"),        options=[0, 1, 2, 3], index=0)
    symptoms_score     = st.slider(fl("symptoms_score"),           min_value=0,    max_value=10,    value=2)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button(CHECK_RISK_BTN):

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
    if systolic_bp > 140:          high_risk_flags.append(f"{sfl('systolic_bp')} {systolic_bp} mmHg > 140")
    if diastolic_bp > 90:          high_risk_flags.append(f"{sfl('diastolic_bp')} {diastolic_bp} mmHg > 90")
    if random_blood_sugar > 200:   high_risk_flags.append(f"{sfl('random_blood_sugar')} {random_blood_sugar} mg/dL > 200")
    if hba1c > 6.5:                high_risk_flags.append(f"{sfl('hba1c')} {hba1c}% > 6.5")
    if hemoglobin < 9.0:           high_risk_flags.append(f"{sfl('hemoglobin')} {hemoglobin} g/dL < 9.0")
    if bmi > 30:                   high_risk_flags.append(f"{sfl('bmi')} {bmi} > 30")
    if spo2 < 92:                  high_risk_flags.append(f"{sfl('spo2')} {spo2}% < 92")
    if age > 40:                   high_risk_flags.append(f"{sfl('age')} {age} {YRS} > 40")
    if edema_severity >= 3:        high_risk_flags.append(f"{sfl('edema_severity')} {edema_severity} ({SEVERE})")

    # ── Mild Risk rules (moderate abnormality) — inclusive boundaries ─────────
    mild_risk_flags = []
    if 130 <= systolic_bp <= 140:          mild_risk_flags.append(f"{sfl('systolic_bp')} {systolic_bp} mmHg (130–140)")
    if 85 <= diastolic_bp <= 90:           mild_risk_flags.append(f"{sfl('diastolic_bp')} {diastolic_bp} mmHg (85–90)")
    if 140 < random_blood_sugar <= 200:    mild_risk_flags.append(f"{sfl('random_blood_sugar')} {random_blood_sugar} mg/dL (140–200)")
    if 5.7 <= hba1c <= 6.5:                mild_risk_flags.append(f"{sfl('hba1c')} {hba1c}% (5.7–6.5)")
    if 9.0 <= hemoglobin <= 11.0:          mild_risk_flags.append(f"{sfl('hemoglobin')} {hemoglobin} g/dL (9–11)")
    if 25 < bmi <= 30:                     mild_risk_flags.append(f"{sfl('bmi')} {bmi} (25–30)")
    if 92 <= spo2 <= 95:                   mild_risk_flags.append(f"{sfl('spo2')} {spo2}% (92–95)")
    if 35 < age <= 40:                     mild_risk_flags.append(f"{sfl('age')} {age} {YRS} (35–40)")
    if edema_severity == 2:                mild_risk_flags.append(f"{sfl('edema_severity')} {edema_severity} ({MODERATE})")
    if symptoms_score >= 6:                mild_risk_flags.append(f"{sfl('symptoms_score')} {symptoms_score}/10")

    # ── Determine risk level ──────────────────────────────────────────────────
    if high_risk_flags or probability >= 0.30 or len(mild_risk_flags) >= 3:
        risk_level = "HIGH"
    elif mild_risk_flags:
        risk_level = "MILD"
    else:
        risk_level = "LOW"

    # ── Save to history ──────────────────────────────────────────────────────
    try:
        save_prediction(st.session_state.username, raw_input, risk_level, float(probability))
        saved_ok = True
    except Exception:
        saved_ok = False

    st.markdown(f'<p class="section-label">{SECTION_RESULT}</p>', unsafe_allow_html=True)

    res_col, xai_col = st.columns([1, 1.6])

    with res_col:
        if risk_level == "HIGH":
            st.markdown(f"""
            <div class="result-high">
                <h2>{RESULT_HIGH_TITLE}</h2>
                <p>{RESULT_HIGH_MSG}</p>
            </div>
            """, unsafe_allow_html=True)
            if high_risk_flags:
                st.markdown(f'<p class="card-title" style="color:#C0392B;letter-spacing:1.5px;font-size:0.72rem;margin-top:16px;">{HIGH_FLAGS_TITLE}</p>', unsafe_allow_html=True)
                for f in high_risk_flags:
                    st.markdown(f'<div class="factor-tag"><strong>↑</strong> {f}</div>', unsafe_allow_html=True)
            if mild_risk_flags:
                st.markdown(f'<p class="card-title" style="color:#B7770D;letter-spacing:1.5px;font-size:0.72rem;margin-top:16px;">{MILD_ALSO_TITLE}</p>', unsafe_allow_html=True)
                for f in mild_risk_flags:
                    st.markdown(f'<div class="factor-tag" style="border-left-color:#E8C43A;background:#FEF9E8;color:#7A5010;"><strong>!</strong> {f}</div>', unsafe_allow_html=True)

        elif risk_level == "MILD":
            st.markdown(f"""
            <div class="result-mild">
                <h2>{RESULT_MILD_TITLE}</h2>
                <p>{RESULT_MILD_MSG}</p>
            </div>
            """, unsafe_allow_html=True)
            if mild_risk_flags:
                st.markdown(f'<p class="card-title" style="color:#B7770D;letter-spacing:1.5px;font-size:0.72rem;margin-top:16px;">{MILD_FLAGS_TITLE}</p>', unsafe_allow_html=True)
                for f in mild_risk_flags:
                    st.markdown(f'<div class="factor-tag" style="border-left-color:#E8C43A;background:#FEF9E8;color:#7A5010;"><strong>!</strong> {f}</div>', unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-safe">
                <h2>{RESULT_LOW_TITLE}</h2>
                <p>{RESULT_LOW_MSG}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="debug-box">
            <b>{QUICK_SUMMARY}</b><br>
            {SERIOUS_CONCERNS}: {len(high_risk_flags)}<br>
            {MINOR_CONCERNS}: {len(mild_risk_flags)}
        </div>
        """, unsafe_allow_html=True)

        if saved_ok:
            st.markdown(f'<p style="color:#2E7D5A; font-size:0.85rem; margin-top:10px;">{SAVED_NOTE}</p>', unsafe_allow_html=True)

    with xai_col:
        st.markdown(f'<div class="card"><p class="card-title">{WHY_RESULT}</p>', unsafe_allow_html=True)

        checks = [
            (systolic_bp,        90,   140,  sfl('systolic_bp'),      'mmHg'),
            (diastolic_bp,       60,   90,   sfl('diastolic_bp'),     'mmHg'),
            (random_blood_sugar, 70,   140,  sfl('random_blood_sugar'), 'mg/dL'),
            (hba1c,              4.0,  5.6,  sfl('hba1c'),            '%'),
            (hemoglobin,         11.0, 15.0, sfl('hemoglobin'),       'g/dL'),
            (bmi,                18.5, 24.9, sfl('bmi'),              ''),
            (spo2,               95,   100,  sfl('spo2'),             '%'),
            (body_temp,          97.0, 99.0, sfl('body_temp'),        '°F'),
            (heart_rate,         60,   100,  sfl('heart_rate'),       'bpm'),
            (respiratory_rate,   12,   20,   sfl('respiratory_rate'), 'bpm'),
            (symptoms_score,     0,    3,    sfl('symptoms_score'),   ''),
            (age,                18,   35,   sfl('age'),              YRS),
        ]

        high_factors   = []
        low_factors    = []
        normal_factors = []

        for val, lo, hi, label, unit in checks:
            u = f" {unit}" if unit else ""
            if val > hi:
                high_factors.append(f"<b>{label}</b> {ABOVE_NORMAL}: {val}{u} ({lo}–{hi}{u})")
            elif val < lo:
                low_factors.append(f"<b>{label}</b> {BELOW_NORMAL}: {val}{u} ({lo}–{hi}{u})")
            else:
                normal_factors.append(f"<b>{label}</b>: {val}{u} — {WITHIN_NORMAL}")

        if high_factors:
            st.markdown(f'<p class="card-title" style="color:#C0622A;font-size:0.72rem;letter-spacing:1.5px;">{ELEVATED_PARAMS}</p>', unsafe_allow_html=True)
            for f in high_factors:
                st.markdown(f'<div class="factor-tag">⬆ {f}</div>', unsafe_allow_html=True)

        if low_factors:
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-title" style="color:#2A5EA0;font-size:0.72rem;letter-spacing:1.5px;">{LOW_PARAMS}</p>', unsafe_allow_html=True)
            for f in low_factors:
                st.markdown(f'<div class="factor-tag" style="border-left-color:#5A8ABF;background:#EEF3FE;color:#1A3060;">⬇ {f}</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(f'<p class="card-title" style="color:#2E7D5A;font-size:0.72rem;letter-spacing:1.5px;">{NORMAL_PARAMS}</p>', unsafe_allow_html=True)
        for f in normal_factors:
            st.markdown(f'<div class="factor-tag" style="border-left-color:#72BFA0;background:#EEF7F2;color:#1A4A30;">✓ {f}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="footer">{FOOTER}</div>', unsafe_allow_html=True)