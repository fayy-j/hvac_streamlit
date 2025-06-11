import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set paths for .pkl files
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/main/hvac_preprocessed.csv"  # change this
    df = pd.read_csv(url)
    df = df.drop(columns=["Timestamp"], errors="ignore")
    return df

df = load_data()

# Features
feature_cols = ['T_Supply', 'T_Return', 'T_Outdoor', 'T_Saturation']
X = df[feature_cols]
y = df["Energy"]

# Load or train model and scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

# UI
st.title("Energy Consumption Prediction")
st.markdown("Enter raw input values to predict energy consumption (no need to standardize).")

user_input = st.text_input(
    "Enter values for: T_Supply, T_Return, T_Outdoor, T_Saturation",
    placeholder="e.g., 20.5, 19.8, 55.2, 60.1"
)

if user_input:
    try:
        input_values = np.array([float(x.strip()) for x in user_input.split(',')])
        if len(input_values) != 4:
            st.error("Please enter exactly 4 values.")
        else:
            input_scaled = scaler.transform(input_values.reshape(1, -1))
            prediction = model.predict(input_scaled)[0]
            st.success(f"Predicted Energy Consumption: {prediction:.4f}")
    except Exception as e:
        st.error(f"Invalid input. Error: {e}")
