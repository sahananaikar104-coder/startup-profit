import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('50_Startups_IndianStates.csv')

# Convert State column to dummies
df = pd.get_dummies(df, columns=['State'], drop_first=True)

# Features and target
X = df.drop('Profit', axis=1)
y = df['Profit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor().fit(X_train, y_train)

# Streamlit UI
st.title("🚀 Startup Profit Prediction (Indian Cities Version)")
st.write("Enter details to predict startup profit")

# Inputs
rd = st.number_input("R&D Spend", min_value=0.0)
admin = st.number_input("Administration Spend", min_value=0.0)
marketing = st.number_input("Marketing Spend", min_value=0.0)

state = st.selectbox("State", ["Bangalore", "Mysuru", "Bagalkote"])

# Prepare input row
input_data = pd.DataFrame({
    'R&D Spend': [rd],
    'Administration': [admin],
    'Marketing Spend': [marketing],
    'State_Mysuru': [1 if state == "Mysuru" else 0],
    'State_Bagalkote': [1 if state == "Bagalkote" else 0]
})

# Predict
if st.button("Predict Profit"):
    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    st.success(f"Linear Regression Profit: ₹{lr_pred:,.0f}")
    st.success(f"Random Forest Profit: ₹{rf_pred:,.0f}")
