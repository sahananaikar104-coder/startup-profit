import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Startup Profit Prediction", layout="centered")

st.title("🚀 Startup Profit Prediction (Indian Cities Version)")
st.write("Enter details below to predict startup profit")

# Load model
model = joblib.load("startup_profit_model.pkl")

# User inputs
rnd = st.number_input("R&D Spend", min_value=0.0, format="%.2f")
admin = st.number_input("Administration Spend", min_value=0.0, format="%.2f")
marketing = st.number_input("Marketing Spend", min_value=0.0, format="%.2f")

state = st.selectbox("State", ["Bangalore", "Mysuru", "Bagalkote"])

# Prepare input data
input_data = pd.DataFrame({
    "R&D Spend": [rnd],
    "Administration": [admin],
    "Marketing Spend": [marketing],
    "State": [state]
})

# Predict
if st.button("Predict Profit"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Profit: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error("⚠️ Error occurred. Check app logs.")
