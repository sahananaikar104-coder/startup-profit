import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv('50_Startups.csv')


# Encode 'State'
df = pd.get_dummies(df, columns=['State'], drop_first=True)

# Split Features and Target
X = df.drop('Profit', axis=1)
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit App Layout
st.title("🚀 Startup Profit Predictor")
st.markdown("Enter startup details below to predict the expected profit:")

# Input widgets
rd_spend = st.number_input("R&D Spend", min_value=0, value=0)
admin_spend = st.number_input("Administration Spend", min_value=0, value=0)
marketing_spend = st.number_input("Marketing Spend", min_value=0, value=0)
state = st.selectbox("State", ["California", "New York", "Florida"])

# Button
if st.button("Predict Profit"):
    # Prepare input data
    input_data = pd.DataFrame(0, index=[0], columns=X_train.columns)
    input_data['R&D Spend'] = rd_spend
    input_data['Administration'] = admin_spend
    input_data['Marketing Spend'] = marketing_spend
    if 'State_California' in input_data.columns:
        input_data['State_California'] = 1 if state=="California" else 0
    if 'State_New York' in input_data.columns:
        input_data['State_New York'] = 1 if state=="New York" else 0
    
    # Predict
    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]
    
    # Display results
    st.subheader("📊 Predicted Profits")
    st.markdown(f"**Linear Regression:** ₹{lr_pred:,.0f}")
    st.markdown(f"**Random Forest:** ₹{rf_pred:,.0f}")
    
    higher = "Linear Regression" if lr_pred > rf_pred else "Random Forest"
    st.markdown(f"**Higher Profit Model:** {higher}")
