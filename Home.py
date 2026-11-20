from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

st.header("โปรเจคการตรวจโรคหัวใจ")

col1, col2 = st.columns(2)

with col1:
    st.header("เป็นโรคหัวใจ")
    st.image("./img/yes.jpg")

with col2:
    st.header("ไม่เป็นโรคหัวใจ")
    st.image("./img/no.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลการเป็นโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

# นับจำนวนผู้ที่เป็นและไม่เป็นโรคหัวใจเพื่อแสดงกราฟแท่ง
heart_counts = dt['HeartDisease'].value_counts().rename(
    index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"}
)

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(heart_counts)
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# รับค่า Feature ทั้ง 11 ตัวให้ตรงกับคอลัมน์ใน Heart3.csv
age = st.slider("อายุ (Age)", min_value=20, max_value=100, value=50)

sex = st.selectbox(
    "เพศ (Sex)",
    options=[1, 0],
    format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)"
)

chest_pain = st.selectbox(
    "ประเภทอาการเจ็บหน้าอก (ChestPainType)",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 - ไม่มีอาการเจ็บหน้าอก (ASY)",
        2: "2 - เจ็บหน้าอกแบบไม่คงที่ (ATA)",
        3: "3 - เจ็บหน้าอกที่ไม่ใช่โรคหัวใจ (NAP)",
        4: "4 - เจ็บหน้าอกแบบทั่วไป (TA)"
    }[x]
)

resting_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=0, max_value=250, value=130)
cholesterol = st.number_input("ระดับคอเลสเตอรอล (Cholesterol)", min_value=0, max_value=700, value=200)

fasting_bs = st.selectbox(
    "น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl (FastingBS)",
    options=[0, 1],
    format_func=lambda x: "ปกติ <= 120 mg/dl (0)" if x == 0 else "สูง > 120 mg/dl (1)"
)

resting_ecg = st.selectbox(
    "คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "1 - กล้ามเนื้อหัวใจห้องล่างซ้ายหนา (LVH)",
        2: "2 - ปกติ (Normal)",
        3: "3 - คลื่น ST-T ผิดปกติ (ST)"
    }[x]
)

max_hr = st.slider("อัตราการเต้นของหัวใจสูงสุด (MaxHR)", min_value=60, max_value=220, value=140)

exercise_angina = st.selectbox(
    "เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)",
    options=[0, 1],
    format_func=lambda x: "ไม่เจ็บ (0)" if x == 0 else "เจ็บ (1)"
)

oldpeak = st.number_input("ค่า Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)

st_slope = st.selectbox(
    "ความชันของคลื่น ST (ST_Slope)",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "1 - ลาดลง (Down)",
        2: "2 - ราบเรียบ (Flat)",
        3: "3 - ลาดขึ้น (Up)"
    }[x]
)

if st.button("ทำนายผล"):
    X = dt.drop('HeartDisease', axis=1)
    y = dt['HeartDisease']

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    # สร้าง DataFrame ที่มีชื่อคอลัมน์ตรงกับ X เพื่อป้องกัน Warning
    x_input = pd.DataFrame(
        [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
          resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
        columns=X.columns
    )

    out = Knn_model.predict(x_input)

    if out[0] == 1:
        st.error("ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (1)")
        st.image("./img/yes.jpg")
    else:
        st.success("ผลการทำนาย: ไม่เป็นโรคหัวใจ (0)")
        st.image("./img/no.jpg")
else:
    st.write("ไม่ทำนาย")