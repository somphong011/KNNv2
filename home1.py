import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageStat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioSense AI | ระบบวิเคราะห์โรคหัวใจ KNN",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. CUSTOM CSS (Black, Navy Blue, Sky Blue, Crisp White)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Prompt', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #060B19 0%, #0D1B3E 50%, #050914 100%);
        color: #FFFFFF !important;
    }

    /* บังคับตัวหนังสือ Label ทุกจุดให้เป็นสีขาวสว่างชัดเจน */
    label, .stMarkdown p, .stRadio label, .stSelectbox label, .stSlider label, .stNumberInput label, [data-testid="stWidgetLabel"] p {
        color: #F8FAFC !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* กล่อง Input สีขาว ตัวอักษรดำ อ่านง่าย */
    input[type="number"], input[type="text"], div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 10px !important;
        border: 2px solid #38BDF8 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }

    /* แถบซ้ายมือ (Sidebar) */
    section[data-testid="stSidebar"] {
        background: #081229 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.3);
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    /* แบนเนอร์ด้านบน */
    .hero-box {
        background: linear-gradient(120deg, #0284C7 0%, #1E40AF 60%, #0F172A 100%);
        border: 1px solid #7DD3FC;
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(2, 132, 199, 0.3);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #E0F2FE;
        margin-top: 4px;
    }

    /* การ์ดตัวเลข KPI */
    .kpi-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #38BDF8;
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #7DD3FC;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 2px;
    }

    /* อนิเมชันหัวใจเต้น */
    @keyframes pulse-ok {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.09); }
    }
    @keyframes pulse-warn {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.15); }
    }
    .anim-ok { animation: pulse-ok 1.5s infinite ease-in-out; display: inline-block; }
    .anim-warn { animation: pulse-warn 0.8s infinite ease-in-out; display: inline-block; }

    /* การ์ดแสดงผลลัพธ์หัวใจ */
    .res-box-ok {
        background: linear-gradient(160deg, #064E3B 0%, #0C4A6E 55%, #0F172A 100%);
        border: 2px solid #34D399;
        border-radius: 20px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 10px 28px rgba(52, 211, 153, 0.25);
    }
    .res-box-warn {
        background: linear-gradient(160deg, #881337 0%, #1E3A8A 60%, #0F172A 100%);
        border: 2px solid #FB7185;
        border-radius: 20px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 10px 28px rgba(251, 113, 133, 0.3);
    }

    /* กล่องวิเคราะห์ภาพถ่าย */
    .photo-box {
        background: rgba(15, 23, 42, 0.88);
        border: 1.5px solid #38BDF8;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* ปุ่มกดวิเคราะห์ */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #0EA5E9 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.45) !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #38BDF8 0%, #3B82F6 100%) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. SVG HEART GRAPHICS
# ---------------------------------------------------------
HEART_OK_SVG = """<div class="anim-ok"><svg width="110" height="110" viewBox="0 0 140 140" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="70" cy="70" r="62" fill="rgba(52, 211, 153, 0.16)" stroke="#34D399" stroke-width="2.5" stroke-dasharray="6 4"/><path d="M70 112C70 112 24 83.5 24 51.5C24 37.4 35.2 26 49 26C57.6 26 65.2 30.4 70 37.2C74.8 30.4 82.4 26 91 26C104.8 26 116 37.4 116 51.5C116 83.5 70 112 70 112Z" fill="#0EA5E9"/><path d="M34 66H52L58 52L66 80L74 46L81 72L86 66H106" stroke="#FFFFFF" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="104" cy="98" r="16" fill="#10B981" stroke="#FFFFFF" stroke-width="2.5"/><path d="M97 98L102 103L112 93" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg></div>"""

HEART_WARN_SVG = """<div class="anim-warn"><svg width="110" height="110" viewBox="0 0 140 140" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="70" cy="70" r="62" fill="rgba(251, 113, 133, 0.18)" stroke="#FB7185" stroke-width="2.5" stroke-dasharray="5 3"/><path d="M70 112C70 112 24 83.5 24 51.5C24 37.4 35.2 26 49 26C57.6 26 65.2 30.4 70 37.2C74.8 30.4 82.4 26 91 26C104.8 26 116 37.4 116 51.5C116 83.5 70 112 70 112Z" fill="#F43F5E"/><path d="M72 42L62 64H75L65 88" stroke="#FEF08A" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="104" cy="98" r="16" fill="#F59E0B" stroke="#FFFFFF" stroke-width="2.5"/><path d="M104 90V99" stroke="#FFFFFF" stroke-width="3.2" stroke-linecap="round"/><circle cx="104" cy="105" r="1.8" fill="#FFFFFF"/></svg></div>"""

# ---------------------------------------------------------
# 4. IMAGE ANALYSIS HELPER (วิเคราะห์ภาพถ่ายเบื้องต้น)
# ---------------------------------------------------------
def analyze_user_photo(pil_img):
    rgb_img = pil_img.convert("RGB")
    stat = ImageStat.Stat(rgb_img)
    r_mean, g_mean, b_mean = stat.mean
    brightness = (0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean) / 255.0 * 100

    # ประเมินโทนสีภาพและแสงสว่าง
    red_ratio = r_mean / (r_mean + g_mean + b_mean + 1e-5) * 100
    if brightness < 30:
        light_status = "แสงน้อย (ภาพค่อนข้างมืด)"
    elif brightness > 85:
        light_status = "แสงจ้าเกินไป"
    else:
        light_status = "แสงเหมาะสมชัดเจน (ดี)"

    if red_ratio > 42:
        tone_note = "โทนสีภาพค่อนข้างแดง/อุ่นสูง"
    elif red_ratio < 31:
        tone_note = "โทนสีภาพค่อนข้างซีด/เย็น"
    else:
        tone_note = "โทนสีผิวและสีภาพอยู่ในเกณฑ์สมดุล"

    return {
        "brightness": brightness,
        "red_ratio": red_ratio,
        "light_status": light_status,
        "tone_note": tone_note,
        "resolution": f"{rgb_img.width} × {rgb_img.height} px"
    }

# ---------------------------------------------------------
# 5. LOAD DATA & TRAIN KNN MODEL
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Heart3.csv")

@st.cache_resource
def build_model(df, k_neighbors):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k_neighbors, weights="distance", metric="euclidean")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test
    }
    return model, scaler, X.columns.tolist(), stats

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ ไม่พบไฟล์ Heart3.csv กรุณานำไฟล์ Heart3.csv มาวางไว้ในโฟลเดอร์เดียวกับ home.py")
    st.stop()

# ---------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫀 CardioSense AI")
    st.markdown("ระบบคัดกรองความเสี่ยงโรคหัวใจอัจฉริยะ")
    st.divider()

    menu = st.radio(
        "📌 เลือกเมนูการใช้งาน",
        options=[
            "🩺 ตรวจประเมินรายบุคคล",
            "📁 ทำนายชุดข้อมูลใหม่ (CSV)",
            "📊 ประสิทธิภาพโมเดล KNN",
            "🗂️ ดูฐานข้อมูล Heart3"
        ]
    )

    st.divider()
    with st.expander("⚙️ ปรับค่า K ของโมเดล (ขั้นสูง)"):
        k_value = st.select_slider("เลือกค่า K (เพื่อนบ้านใกล้เคียง)", options=[3, 5, 7, 9, 11, 13, 15], value=7)
        st.caption("ค่าแนะนำคือ K = 7 (แม่นยำสูงสุด)")

model, scaler, feature_names, stats = build_model(df, k_value)

# ---------------------------------------------------------
# 7. TOP BANNER & KPI SUMMARY
# ---------------------------------------------------------
st.markdown("""
<div class="hero-box">
    <h1 class="hero-title">🫀 ระบบวิเคราะห์และพยากรณ์ความเสี่ยงโรคหัวใจ (KNN)</h1>
    <p class="hero-subtitle">รองรับการแนบภาพถ่ายผู้รับการตรวจร่วมกับข้อมูลสุขภาพ 11 ตัวแปร ประมวลผลแม่นยำด้วยอัลกอริทึม KNN</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">🎯 ความแม่นยำของระบบ (Accuracy)</div><div class="kpi-value">{stats["accuracy"]*100:.2f}% (K={k_value})</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">⚡ ความไวในการตรวจพบความเสี่ยง</div><div class="kpi-value">{stats["recall"]*100:.2f}%</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">👥 จำนวนข้อมูลอ้างอิงในระบบ</div><div class="kpi-value">{len(df):,} ราย</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 8. PAGE CONTENT ROUTING
# ---------------------------------------------------------

# === PAGE 1: ตรวจประเมินรายบุคคล + ใส่ภาพตัวเอง ===
if menu == "🩺 ตรวจประเมินรายบุคคล":
    col_left, col_right = st.columns([1.35, 1], gap="large")

    with col_left:
        st.markdown("### 📸 1. ใส่ภาพถ่ายผู้รับการตรวจ (ไม่บังคับ)")
        photo_mode = st.radio(
            "เลือกรูปแบบการใส่ภาพ",
            options=["อัปโหลดรูปภาพจากเครื่อง", "ถ่ายภาพจากกล้องเว็บแคม"],
            horizontal=True
        )

        user_image_file = None
        if photo_mode == "อัปโหลดรูปภาพจากเครื่อง":
            user_image_file = st.file_uploader("เลือกรูปภาพของคุณ (.jpg, .png)", type=["jpg", "jpeg", "png"])
        else:
            user_image_file = st.camera_input("กดปุ่มถ่ายภาพด้านล่าง")

        st.divider()
        st.markdown("### 📝 2. กรอกข้อมูลสุขภาพผู้รับการตรวจ")
        with st.form("single_predict_form"):
            patient_name = st.text_input("ชื่อ-นามสกุล ผู้รับการตรวจ", value="ผู้รับการตรวจทั่วไป")

            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=50)
            with c2:
                sex = st.selectbox("เพศ", options=[1, 0], format_func=lambda x: "👨 ชาย (1)" if x == 1 else "👩 หญิง (0)")

            c3, c4 = st.columns(2)
            with c3:
                resting_bp = st.number_input("ความดันโลหิตขณะพัก (mm Hg)", min_value=60, max_value=240, value=125)
            with c4:
                cholesterol = st.number_input("ระดับคอเลสเตอรอล (mg/dl)", min_value=0, max_value=650, value=210)

            c5, c6 = st.columns(2)
            with c5:
                max_hr = st.number_input("ชีพจรสูงสุดขณะออกกำลัง (MaxHR)", min_value=60, max_value=220, value=150)
            with c6:
                fasting_bs = st.selectbox("น้ำตาลในเลือดหลังอดอาหาร > 120", options=[0, 1], format_func=lambda x: "✅ ไม่เกิน 120 mg/dl (0)" if x == 0 else "⚠️ เกิน 120 mg/dl (1)")

            c7, c8 = st.columns(2)
            with c7:
                chest_pain = st.selectbox(
                    "ลักษณะอาการเจ็บหน้าอก",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "1 - ไม่มีอาการเจ็บหน้าอก (ASY)",
                        2: "2 - เจ็บหน้าอกไม่ตรงแบบ (ATA)",
                        3: "3 - เจ็บที่ไม่ใช่โรคหัวใจ (NAP)",
                        4: "4 - เจ็บหน้าอกแบบตรงแบบ (TA)"
                    }[x]
                )
            with c8:
                exercise_angina = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย", options=[0, 1], format_func=lambda x: "ไม่มีอาการ (0)" if x == 0 else "มีอาการเจ็บ (1)")

            c9, c10, c11 = st.columns(3)
            with c9:
                resting_ecg = st.selectbox(
                    "คลื่นไฟฟ้าหัวใจ (ECG)",
                    options=[2, 1, 3],
                    format_func=lambda x: {2: "2 - ปกติ", 1: "1 - กล้ามเนื้อหนา", 3: "3 - ผิดปกติ (ST)"}[x]
                )
            with c10:
                st_slope = st.selectbox(
                    "ความชันคลื่น ST",
                    options=[3, 2, 1],
                    format_func=lambda x: {3: "3 - ลาดขึ้น (Up)", 2: "2 - แบนราบ (Flat)", 1: "1 - ลาดลง (Down)"}[x]
                )
            with c11:
                oldpeak = st.number_input("ค่า Oldpeak", min_value=-2.5, max_value=6.5, value=0.0, step=0.1)

            st.markdown("<br>", unsafe_allow_html=True)
            st.form_submit_button("💙 วิเคราะห์ภาพและประเมินความเสี่ยงทันที", use_container_width=True)

    with col_right:
        st.markdown("### 🫀 ผลการวิเคราะห์สุขภาพหัวใจ")

        # แสดงภาพถ่ายและผลการวิเคราะห์ภาพ (ถ้ามีการอัปโหลด/ถ่ายภาพ)
        if user_image_file is not None:
            pil_img = Image.open(user_image_file)
            photo_info = analyze_user_photo(pil_img)

            st.markdown('<div class="photo-box">', unsafe_allow_html=True)
            p_col1, p_col2 = st.columns([1, 1.4])
            with p_col1:
                st.image(pil_img, caption=f"คุณ {patient_name}", use_container_width=True)
            with p_col2:
                st.markdown(f"**👤 ผู้รับการตรวจ:** {patient_name}")
                st.markdown(f"**📷 ความละเอียดภาพ:** {photo_info['resolution']}")
                st.markdown(f"**💡 สภาพแสง:** {photo_info['light_status']} ({photo_info['brightness']:.0f}%)")
                st.markdown(f"**🎨 ผลวิเคราะห์โทนภาพ:** {photo_info['tone_note']}")
                st.caption("หมายเหตุ: วิเคราะห์คุณภาพภาพเบื้องต้นประกอบรายงานผล")
            st.markdown('</div>', unsafe_allow_html=True)

        # คำนวณผลลัพธ์จากโมเดล KNN
        input_data = pd.DataFrame([{
            "Age": age, "Sex": sex, "ChestPainType": chest_pain,
            "RestingBP": resting_bp, "Cholesterol": cholesterol, "FastingBS": fasting_bs,
            "RestingECG": resting_ecg, "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope
        }])

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        risk_pct = proba[1] * 100
        safe_pct = proba[0] * 100

        if pred == 1:
            html_result = f"""
<div class="res-box-warn">
{HEART_WARN_SVG}
<p style="color: #BAE6FD; font-size: 0.95rem; margin: 6px 0 0 0;">ผู้รับการตรวจ: <b>{patient_name}</b></p>
<h2 style="color: #FDA4AF; margin: 4px 0 6px 0; font-size: 1.55rem;">พบความเสี่ยงเป็นโรคหัวใจ</h2>
<p style="color: #FFFFFF; font-size: 1.05rem; margin-bottom: 12px;">แนวโน้มความเสี่ยง <b style="color: #FB7185; font-size: 1.3rem;">{risk_pct:.1f}%</b></p>
<div style="background: rgba(0,0,0,0.35); border-radius: 10px; padding: 12px; font-size: 0.9rem; color: #F8FAFC;">
💡 <b>คำแนะนำ:</b> ค่าสุขภาพมีความใกล้เคียงกับกลุ่มเสี่ยง แนะนำให้พบแพทย์เพื่อตรวจคลื่นไฟฟ้าหัวใจเพิ่มเติม
</div>
</div>
"""
            st.markdown(html_result, unsafe_allow_html=True)
        else:
            html_result = f"""
<div class="res-box-ok">
{HEART_OK_SVG}
<p style="color: #BAE6FD; font-size: 0.95rem; margin: 6px 0 0 0;">ผู้รับการตรวจ: <b>{patient_name}</b></p>
<h2 style="color: #6EE7B7; margin: 4px 0 6px 0; font-size: 1.55rem;">ไม่เป็นโรคหัวใจ (ปกติ)</h2>
<p style="color: #FFFFFF; font-size: 1.05rem; margin-bottom: 12px;">โอกาสหัวใจแข็งแรงปกติ <b style="color: #34D399; font-size: 1.3rem;">{safe_pct:.1f}%</b></p>
<div style="background: rgba(0,0,0,0.35); border-radius: 10px; padding: 12px; font-size: 0.9rem; color: #F8FAFC;">
✨ <b>คำแนะนำ:</b> ผลประเมินอยู่ในเกณฑ์ปกติ ควรออกกำลังกายสม่ำเสมอและตรวจสุขภาพประจำปี
</div>
</div>
"""
            st.markdown(html_result, unsafe_allow_html=True)

        # กราฟหน้าปัดความเสี่ยง
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 34}},
            title={"text": "ดัชนีความเสี่ยง (Risk Score)", "font": {"color": "#BAE6FD", "size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#BAE6FD"},
                "bar": {"color": "#FB7185" if pred == 1 else "#34D399"},
                "bgcolor": "#0F172A",
                "bordercolor": "#38BDF8",
                "steps": [
                    {"range": [0, 40], "color": "rgba(52, 211, 153, 0.2)"},
                    {"range": [40, 70], "color": "rgba(56, 189, 248, 0.25)"},
                    {"range": [70, 100], "color": "rgba(251, 113, 133, 0.28)"}
                ]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

# === PAGE 2: ทำนายชุดข้อมูลใหม่ (CSV) ===
elif menu == "📁 ทำนายชุดข้อมูลใหม่ (CSV)":
    st.markdown("### 📁 อัปโหลดไฟล์ CSV เพื่อทำนายผลทีละหลายรายการ")
    st.info("ไฟล์ CSV ต้องมีคอลัมน์คุณลักษณะครบทั้ง 11 คอลัมน์ (สามารถกดดาวน์โหลดไฟล์ตัวอย่างไปลองแก้ไขข้อมูลได้เลย)")

    template_csv = df.drop("HeartDisease", axis=1).head(10).to_csv(index=False).encode("utf-8")
    st.download_button("📥 ดาวน์โหลดไฟล์ตัวอย่าง (Template CSV)", data=template_csv, file_name="sample_patients.csv", mime="text/csv")

    uploaded = st.file_uploader("เลือกไฟล์ CSV ที่ต้องการทำนาย", type=["csv"])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        missing = [c for c in feature_names if c not in new_df.columns]
        if missing:
            st.error(f"ไฟล์ขาดคอลัมน์: {', '.join(missing)}")
        else:
            X_new_scaled = scaler.transform(new_df[feature_names])
            preds = model.predict(X_new_scaled)
            probs = model.predict_proba(X_new_scaled)[:, 1] * 100

            out_df = new_df.copy()
            out_df["ผลการทำนาย"] = np.where(preds == 1, "❤️‍🩹 เสี่ยงเป็นโรคหัวใจ (1)", "💙 ไม่เป็นโรคหัวใจ (0)")
            out_df["ความเสี่ยง (%)"] = np.round(probs, 2)

            c1, c2, c3 = st.columns(3)
            c1.metric("จำนวนทั้งหมด", f"{len(out_df)} ราย")
            c2.metric("พบความเสี่ยง", f"{(preds == 1).sum()} ราย")
            c3.metric("ปกติ", f"{(preds == 0).sum()} ราย")

            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "💾 ดาวน์โหลดผลลัพธ์การทำนาย (.CSV)",
                data=out_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="prediction_results.csv",
                mime="text/csv"
            )

# === PAGE 3: ประสิทธิภาพโมเดล KNN ===
elif menu == "📊 ประสิทธิภาพโมเดล KNN":
    st.markdown("### 📊 กราฟประเมินความแม่นยำของโมเดล KNN")
    col1, col2 = st.columns(2)

    with col1:
        fig_cm = px.imshow(
            stats["cm"], text_auto=True,
            x=["ไม่เป็นโรค (0)", "เป็นโรคหัวใจ (1)"],
            y=["ไม่เป็นโรค (0)", "เป็นโรคหัวใจ (1)"],
            labels=dict(x="ค่าที่โมเดลทำนาย", y="ค่าจริง", color="จำนวน"),
            color_continuous_scale=["#081229", "#1D4ED8", "#0EA5E9", "#7DD3FC"],
            title="Confusion Matrix (ตารางความแม่นยำ)"
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
        acc_list = []
        for k_i in k_list:
            m_tmp = KNeighborsClassifier(n_neighbors=k_i, weights="distance")
            m_tmp.fit(stats["X_train_scaled"], stats["y_train"])
            acc_list.append(accuracy_score(stats["y_test"], m_tmp.predict(stats["X_test_scaled"])) * 100)

        fig_k = go.Figure(go.Scatter(x=k_list, y=acc_list, mode="lines+markers", line=dict(color="#38BDF8", width=3)))
        fig_k.update_layout(
            title="เปรียบเทียบความแม่นยำตามค่า K",
            xaxis_title="ค่า K", yaxis_title="ความแม่นยำ (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#FFFFFF")
        )
        st.plotly_chart(fig_k, use_container_width=True)

# === PAGE 4: ดูฐานข้อมูล Heart3 ===
elif menu == "🗂️ ดูฐานข้อมูล Heart3":
    st.markdown("### 🗂️ ฐานข้อมูลผู้ป่วย `Heart3.csv` (918 รายการ)")
    st.dataframe(df, use_container_width=True, height=450)