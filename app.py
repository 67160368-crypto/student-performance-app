import streamlit as st
import pandas as pd
import joblib

# ===============================
# โหลดโมเดล
# ===============================
model = joblib.load("model_artifacts/student_model.pkl")

# ===============================
# CONFIG PAGE
# ===============================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Prompt:wght@400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
}

/* ── Hide Streamlit default header/footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page background ── */
.stApp {
    background: #F7F8FF !important;
}
.block-container {
    padding: 0 !important;
    max-width: 780px !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #6C63FF 0%, #A78BFA 60%, #43D9A2 100%);
    padding: 2.2rem 2rem 2.5rem;
    text-align: center;
    border-radius: 0 0 28px 28px;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
}
.hero-banner h1 {
    font-family: 'Prompt', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 600;
    color: #fff !important;
    margin-bottom: 4px;
}
.hero-banner p {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.85);
    margin-bottom: 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 5px 16px;
    font-size: 0.78rem;
    color: #fff;
    font-weight: 700;
    letter-spacing: 0.4px;
}

/* ── Info box ── */
.info-box {
    background: #EEF0FF;
    border-left: 4px solid #6C63FF;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: #4A47A3;
    font-weight: 700;
    margin: 0 1.2rem 1rem;
}

/* ── Section header ── */
.section-hdr {
    font-family: 'Prompt', sans-serif !important;
    font-size: 1rem;
    font-weight: 600;
    color: #2D2D4E;
    margin: 1.2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Slider labels ── */
.stSlider > label {
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    color: #8A8FB0 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Slider track ── */
.stSlider [data-baseweb="slider"] {
    margin-top: 4px;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #6C63FF !important;
    border: 2px solid #fff !important;
    box-shadow: 0 1px 8px rgba(108,99,255,0.45) !important;
    width: 20px !important;
    height: 20px !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] div {
    color: #8A8FB0 !important;
    font-size: 0.72rem !important;
    font-weight: 700;
}

/* ── Selectbox ── */
.stSelectbox > label {
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    color: #8A8FB0 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.stSelectbox [data-baseweb="select"] {
    border-radius: 12px !important;
    border: 1.5px solid #E2E4F0 !important;
    background: #fff !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(90deg, #6C63FF, #A78BFA) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.9rem 1.5rem !important;
    font-family: 'Prompt', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 18px rgba(108,99,255,0.38) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    margin-top: 0.6rem;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,99,255,0.48) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ── Result: Grade badge ── */
.grade-badge {
    background: linear-gradient(135deg, #F0FAFA, #F0EEFF);
    border: 2px solid #A78BFA;
    border-radius: 20px;
    padding: 1.4rem 1rem;
    text-align: center;
    margin: 0.8rem 0;
}
.grade-badge .label {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1.2px;
    color: #8A8FB0;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.grade-badge .grade {
    font-family: 'Prompt', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #43D9A2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.05;
}
.grade-badge .sublabel {
    font-size: 0.82rem;
    color: #8A8FB0;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Probability cards ── */
.prob-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1rem;
}
.prob-card {
    flex: 1;
    background: #fff;
    border-radius: 14px;
    border: 1.5px solid #E2E4F0;
    padding: 12px 8px;
    text-align: center;
}
.prob-card .g {
    font-size: 1.3rem;
    font-weight: 800;
    color: #2D2D4E;
}
.prob-card .p {
    font-size: 0.85rem;
    font-weight: 800;
    color: #6C63FF;
    margin: 2px 0 6px;
}
.prob-bar {
    height: 5px;
    border-radius: 5px;
    background: #E2E4F0;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 5px;
}
.fill-a { background: linear-gradient(90deg,#43D9A2,#6C63FF); }
.fill-b { background: linear-gradient(90deg,#6C63FF,#A78BFA); }
.fill-c { background: linear-gradient(90deg,#FFB347,#FF6584); }
.fill-f { background: linear-gradient(90deg,#FF6584,#FF4B6E); }

/* ── Dataframe styling ── */
.stDataFrame {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1.5px solid #E2E4F0 !important;
}

/* ── Divider ── */
hr { border-color: #E2E4F0 !important; }

/* ── Footer info section ── */
.footer-box {
    background: #fff;
    border-radius: 18px;
    border: 1.5px solid #E2E4F0;
    padding: 1.2rem 1.4rem;
    margin-top: 0.5rem;
}
.footer-box h4 {
    font-family: 'Prompt', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #2D2D4E;
    margin-bottom: 10px;
}
.footer-box ul {
    padding-left: 1.2rem;
    color: #8A8FB0;
    font-size: 0.85rem;
    font-weight: 600;
    line-height: 1.8;
}
.footer-caption {
    text-align: center;
    font-size: 0.75rem;
    color: #B0B4CC;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 0.8rem;
    padding-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HERO BANNER
# ===============================
st.markdown("""
<div class="hero-banner">
  <div style="font-size:3rem;margin-bottom:6px;">🎓</div>
  <h1>Student Performance Predictor</h1>
  <p>ทำนายผลการเรียนจากพฤติกรรมของนักเรียน</p>
  <span class="hero-badge">🤖 Random Forest Model &nbsp;|&nbsp; 14,003 records</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="info-box">💡 กรุณากรอกข้อมูลให้ใกล้เคียงความเป็นจริง เพื่อผลลัพธ์ที่แม่นยำ</div>', unsafe_allow_html=True)

# ===============================
# INPUT SECTION
# ===============================
st.markdown('<div class="section-hdr">📊 ข้อมูลพฤติกรรมนักเรียน</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    StudyHours = st.slider("📚 ชั่วโมงอ่านหนังสือ / สัปดาห์", 5, 44, 20)
    Attendance = st.slider("🏫 การเข้าเรียน (%)", 60, 100, 80)
    AssignmentCompletion = st.slider("📝 การส่งงาน (%)", 50, 100, 75)
    OnlineCourses = st.slider("💻 จำนวนคอร์สออนไลน์", 0, 20, 3)

    stress_dict = {"ต่ำ": 0, "ปานกลาง": 1, "สูง": 2}
    StressLevel_label = st.selectbox("😰 ระดับความเครียด", list(stress_dict.keys()))
    StressLevel = stress_dict[StressLevel_label]

with col2:
    binary_dict = {"ไม่มี": 0, "มี": 1}

    Discussions_label = st.selectbox("💬 การมีส่วนร่วมในชั้นเรียน", list(binary_dict.keys()))
    Discussions = binary_dict[Discussions_label]

    Extracurricular_label = st.selectbox("🎯 กิจกรรมเสริม", list(binary_dict.keys()))
    Extracurricular = binary_dict[Extracurricular_label]

    EduTech_label = st.selectbox("📱 ใช้เทคโนโลยีการศึกษา", list(binary_dict.keys()))
    EduTech = binary_dict[EduTech_label]

    gender_dict = {"ชาย": 0, "หญิง": 1}
    Gender_label = st.selectbox("👤 เพศ", list(gender_dict.keys()))
    Gender = gender_dict[Gender_label]

    learning_dict = {
        "👁 Visual (การมองดู)": 0,
        "👂 Auditory (การฟัง)": 1,
        "📖 Reading/Writing (การอ่าน/เขียน)": 2,
        "🤲 Kinesthetic (การลงมือทำ)": 3
    }
    LearningStyle_label = st.selectbox("🧠 Learning Style", list(learning_dict.keys()))
    LearningStyle = learning_dict[LearningStyle_label]

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("🔮  Predict Grade"):

    new_data = pd.DataFrame([{
        'Discussions': Discussions,
        'StressLevel': StressLevel,
        'AssignmentCompletion': AssignmentCompletion,
        'Gender': Gender,
        'OnlineCourses': OnlineCourses,
        'LearningStyle': LearningStyle,
        'EduTech': EduTech,
        'Attendance': Attendance,
        'Extracurricular': Extracurricular,
        'StudyHours': StudyHours
    }])

    pred = model.predict(new_data)
    proba = model.predict_proba(new_data)[0]
    grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'F'}
    grade_desc = {'A': 'ยอดเยี่ยม!', 'B': 'ดีมาก', 'C': 'พอใช้ได้', 'F': 'ต้องปรับปรุง'}
    g = grade_map[pred[0]]
    prob_percent = proba * 100

    # ── Grade Badge ──
    st.markdown(f"""
    <div class="grade-badge">
      <div class="label">🎯 Predicted Grade</div>
      <div class="grade">{g}</div>
      <div class="sublabel">{grade_desc[g]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Cards ──
    st.markdown('<div style="font-size:0.78rem;font-weight:800;letter-spacing:1px;color:#8A8FB0;text-transform:uppercase;text-align:center;margin:0.5rem 0 0.6rem;">ความน่าจะเป็นของแต่ละเกรด (%)</div>', unsafe_allow_html=True)

    grades_info = [
        ("A", prob_percent[0], "fill-a"),
        ("B", prob_percent[1], "fill-b"),
        ("C", prob_percent[2], "fill-c"),
        ("F", prob_percent[3], "fill-f"),
    ]

    cards_html = '<div class="prob-row">'
    for grade, pct, cls in grades_info:
        cards_html += f"""
        <div class="prob-card">
          <div class="g">{grade}</div>
          <div class="p">{pct:.1f}%</div>
          <div class="prob-bar"><div class="prob-fill {cls}" style="width:{pct:.1f}%"></div></div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

st.markdown("---")

# ===============================
# FOOTER INFO
# ===============================
st.markdown("""
<div class="footer-box">
  <h4>📚 ข้อมูลที่ใช้ในโมเดล</h4>
  <ul>
    <li>ใช้ข้อมูลนักเรียนจำนวน 14,003 คน</li>
    <li>ใช้พฤติกรรม เช่น การเข้าเรียน การส่งงาน และการมีส่วนร่วม</li>
    <li>ไม่ใช้ ExamScore เพื่อลด Data Leakage</li>
  </ul>
</div>
<div class="footer-caption">Model: Random Forest &nbsp;|&nbsp; Dataset: student_performance.csv</div>
""", unsafe_allow_html=True)