import streamlit as st
import pickle
import numpy as np

model = pickle.load(
open('model_cancel.pkl','rb')
)

scaler = pickle.load(
open('scaler.pkl','rb')
)

st.set_page_config(
page_title="Prediksi Pembatalan Kamar",
page_icon="🏨",
layout="wide"
)

st.title("🏨 Prediksi Pembatalan Kamar Hotel")

st.markdown("""
Aplikasi Machine Learning untuk memprediksi
apakah customer akan melakukan pembatalan booking kamar hotel atau tidak.
""")

lead_time = st.number_input(
"Lead Time",
min_value=0
)

adults = st.number_input(
"Jumlah Dewasa",
min_value=1
)

children = st.number_input(
"Jumlah Anak",
min_value=0
)

babies = st.number_input(
"Jumlah Bayi",
min_value=0
)

weekend = st.number_input(
"Menginap Weekend",
min_value=0
)

weekday = st.number_input(
"Menginap Weekday",
min_value=0
)

prev_cancel = st.number_input(
"Riwayat Pembatalan",
min_value=0
)

booking_changes = st.number_input(
"Jumlah Perubahan Booking",
min_value=0
)

adr = st.number_input(
"Average Daily Rate"
)

hotel = st.selectbox(
"Tipe Hotel",
[
"City Hotel",
"Resort Hotel"
]
)

meal = st.selectbox(
"Paket Makan",
[
"BB",
"HB",
"FB",
"SC"
]
)

market = st.selectbox(
"Market Segment",
[
"Online",
"Offline",
"Corporate",
"Direct"
]
)

hotel_map = {
"City Hotel":0,
"Resort Hotel":1
}

meal_map = {
"BB":0,
"HB":1,
"FB":2,
"SC":3
}

market_map = {
"Online":0,
"Offline":1,
"Corporate":2,
"Direct":3
}

if st.button("Prediksi"):
    data = np.array([[
    lead_time,
    prev_cancel,
    booking_changes,
    hotel_map[hotel],
    market_map[market]
    ]])

    data_scaled = scaler.transform(data)

    hasil = model.predict(data_scaled)

    if hasil[0] == 1:
        st.error(
        "❌ Customer Berpotensi Membatalkan Kamar"
        )
    else:
        st.success(
        "✅ Customer Tidak Membatalkan Kamar"
        )

st.markdown("""
<style>
.main{
background-color:#F5F7FA;
}

.stButton>button{
background-color:#005BAC;
color:white;
height:50px;
width:100%;
font-size:18px;
border-radius:10px;
}

h1{
color:#005BAC;
text-align:center;
}
</style>
""",unsafe_allow_html=True)
