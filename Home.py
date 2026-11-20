from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบทำนายความเสี่ยงโรคหัวใจ", page_icon="❤️", layout="centered")

st.header("❤️ โปรเจคการจำแนกและทำนายความเสี่ยงโรคหัวใจ")

# แบ่งเป็น 2 คอลัมน์ (ไม่เป็นโรคหัวใจ vs เป็นโรคหัวใจ)
col1, col2 = st.columns(2)

with col1:
   st.subheader("✅ ไม่เป็นโรคหัวใจ (Normal)")
   st.image("https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?auto=format&fit=crop&w=600&q=80", caption="สุขภาพหัวใจแข็งแรง")

with col2:
   st.subheader("⚠️ เป็นโรคหัวใจ (Heart Disease)")
   st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?auto=format&fit=crop&w=600&q=80", caption="มีความเสี่ยงโรคหัวใจ")

st.markdown("---")

# ส่วนที่ 1: สถิติข้อมูล
html_stats = """
<div style="background-color:#EC7063;padding:12px;border-radius:12px;border:2px solid #922B21;color:white;">
<center><h4 style="margin:0;">📊 สถิติข้อมูลผู้ป่วยโรคหัวใจ</h4></center>
</div>
"""
st.markdown(html_stats, unsafe_allow_html=True)
st.markdown("")

# โหลดไฟล์ข้อมูล (อย่าลืมเปลี่ยนชื่อไฟล์ให้ตรงกับไฟล์ CSV ของคุณ เช่น heart.csv)
try:
    dt = pd.read_csv("./data/heart.csv")
    st.write("ตัวอย่างข้อมูล 10 แถวแรก:")
    st.dataframe(dt.head(10), use_container_width=True)

    # ปรับชื่อคอลัมน์ให้ตรงกับข้อมูลโรคหัวใจ (ตัวอย่าง: age, trestbps, chol, thalach)
    # *หมายเหตุ: หากไฟล์ CSV ใช้ชื่อคอลัมน์อื่น สามารถแก้ชื่อในวงเล็บด้านล่างได้เลย*
    dt1 = dt['age'].mean()       # อายุเฉลี่ย
    dt2 = dt['trestbps'].mean()  # ความดันโลหิตเฉลี่ย
    dt3 = dt['chol'].mean()      # คอเลสเตอรอลเฉลี่ย
    dt4 = dt['thalach'].mean()   # อัตราการเต้นหัวใจสูงสุดเฉลี่ย

    dx = pd.DataFrame({
        "ค่าเฉลี่ย": [dt1, dt2, dt3, dt4]
    }, index=["อายุ (Age)", "ความดัน (BP)", "คอเลสเตอรอล (Chol)", "ชีพจรสูงสุด (Max HR)"])

    if st.button("📈 แสดงกราฟสถิติข้อมูล"):
       st.bar_chart(dx)
       if st.button("❌ ซ่อนกราฟ"):
           pass
    else:
        st.caption("กดปุ่มด้านบนเพื่อแสดงการจินตทัศน์ข้อมูล")

except FileNotFoundError:
    st.error("⚠️ ไม่พบไฟล์ './data/heart.csv' กรุณาตรวจสอบโฟลเดอร์ data และชื่อไฟล์ของคุณ")

st.markdown("---")

# ส่วนที่ 2: ทำนายผล
html_predict = """
<div style="background-color:#48C9B0;padding:12px;border-radius:12px;border:2px solid #117864;color:white;">
<center><h4 style="margin:0;">🩺 ระบบทำนายความเสี่ยงโรคหัวใจ</h4></center>
</div>
"""
st.markdown(html_predict, unsafe_allow_html=True)
st.markdown("")

# รับค่า Input (ปรับให้เข้ากับบริบทโรคหัวใจ)
col_in1, col_in2 = st.columns(2)

with col_in1:
    age = st.slider("กรุณาเลือกอายุ (ปี)", min_value=18, max_value=100, value=45)
    trestbps = st.slider("ความดันโลหิตขณะพัก (mm Hg)", min_value=80, max_value=220, value=120)

with col_in2:
    chol = st.number_input("ระดับคอเลสเตอรอล (mg/dl)", min_value=100, max_value=600, value=200)
    thalach = st.number_input("อัตราการเต้นหัวใจสูงสุด (bpm)", min_value=60, max_value=220, value=150)

st.markdown("")

if st.button("🔍 ทำนายผล", use_container_width=True):
   if 'dt' in locals():
       # เตรียมข้อมูลสอนโมเดล (เลือกเฉพาะ 4 ฟีเจอร์หลักให้ตรงกับ Input ด้านบน)
       features = ['age', 'trestbps', 'chol', 'thalach']
       X = dt[features]
       y = dt['target']   # คอลัมน์ผลลัพธ์ (0 = ไม่เป็นโรค, 1 = เป็นโรค)
       
       Knn_model = KNeighborsClassifier(n_neighbors=3)
       Knn_model.fit(X, y)  

       x_input = np.array([[age, trestbps, chol, thalach]])
       out = Knn_model.predict(x_input)

       st.markdown("### ผลการทำนาย:")
       if out[0] == 0:
           st.success("🎉 ผลลัพธ์: **ไม่เป็นโรคหัวใจ (Normal)**")
           st.image("https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?auto=format&fit=crop&w=500&q=80", width=350)
       else:
           st.error("🚨 ผลลัพธ์: **มีความเสี่ยงเป็นโรคหัวใจ (Heart Disease)**")
           st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?auto=format&fit=crop&w=500&q=80", width=350)
   else:
       st.warning("กรุณาอัปโหลดไฟล์ข้อมูลก่อนทำการทำนาย")
else:
    st.info("กรอกข้อมูลด้านบนให้ครบถ้วน แล้วกดปุ่ม 'ทำนายผล'")
