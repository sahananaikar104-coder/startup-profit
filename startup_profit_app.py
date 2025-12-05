import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("50_Startups_IndianStates.csv")

# Create dummy variables
df = pd.get_dummies(df, columns=["State"], drop_first=True)

# Train-test split
X = df.drop("Profit", axis=1)
y = df["Profit"]

X_train, X_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor().fit(X_train, y_train)

# Streamlit UI
st.title("🚀 Startup Profit Prediction (Indian Cities Version)")
st.subheader("Enter details to predict startup profit")

rd = st.number_input("R&D Spend", min_value=0.0)
admin = st.number_input("Administration Spend", min_value=0.0)
marketing = st.number_input("Marketing Spend", min_value=0.0)

# Extract real city names from dataset
original_states = sorted(df.filter(like="State_").columns)
cities = ["Bangalore", "Mysuru", "Bagalkote"]  # Your real state values

state = st.selectbox("State", cities)

# Create input frame
input_data = pd.DataFrame({
    "R&D Spend": [rd],
    "Administration": [admin],
    "Marketing Spend": [marketing]
})

# Add dummy columns exactly like training data
for col in X.columns:
    if col.startswith("State_"):
        city_name = col.split("_")[1]           # extract city
        input_data[col] = 1 if state == city_name else 0

# Predict button
if st.button("Predict Profit"):
    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    st.success(f"Linear Regression Predicted Profit: ₹{lr_pred:,.0f}")
    st.success(f"Random Forest Predicted Profit: ₹{rf_pred:,.0f}")
