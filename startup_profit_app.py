import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Startup Profit Predictor", layout="centered")

st.title("📊 Startup Profit Predictor ")
st.write("Predict profit based on R&D, Administration, Marketing Spend and Location.")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("50_Startups_IndianStates.csv")
    return df

df = load_data()

# ----------------------------
# Preprocessing
# ----------------------------
# One-hot encode the State column
df_encoded = pd.get_dummies(df, columns=["State"], drop_first=True)

# Feature selection
X = df_encoded.drop("Profit", axis=1)
y = df_encoded["Profit"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Get columns for prediction input
model_columns = X_train.columns

# ----------------------------
# User Input Section
# ----------------------------
st.subheader("Enter Startup Details")

rd_spend = st.number_input("R&D Spend", min_value=0.0, step=1000.0)
admin_spend = st.number_input("Administration Spend", min_value=0.0, step=1000.0)
marketing_spend = st.number_input("Marketing Spend", min_value=0.0, step=1000.0)

state = st.selectbox("Select Location", ["Bangalore", "Mysuru", "Bagalkote"])

# ----------------------------
# Prepare Input Data for Model
# ----------------------------
import numpy as np

input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

# Fill numeric values
input_df["R&D Spend"] = rd_spend
input_df["Administration"] = admin_spend
input_df["Marketing Spend"] = marketing_spend

# Handle dummy variables
# Your dataset creates dummy columns like: State_Bagalkote, State_Bangalore, State_Mysuru
for col in model_columns:
    if col.startswith("State_"):
        city_name = col.replace("State_", "")
        input_df[col] = 1 if state == city_name else 0

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Profit"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 **Predicted Profit: ₹{prediction:,.2f}**")

# ----------------------------
# Show Dataset (Optional)
# ----------------------------
with st.expander("Show Dataset"):
    st.dataframe(df)

