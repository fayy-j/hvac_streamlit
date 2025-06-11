import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# GitHub CSV raw URL (replace with your actual one)
CSV_URL = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/hvac_preprocessed.csv"

# Load data
@st.cache_data
def load_and_train_model():
    df = pd.read_csv(CSV_URL)
    
    # Drop Timestamp column if exists
    df = df.drop(columns=["Timestamp"], errors="ignore")

    # Define features and target
    feature_cols = ['T_Supply', 'T_Return', 'T_Outdoor', 'T_Saturation']  # adjust if needed
    target_col = "Energy"
    
    X = df[feature_cols]
    y = df[target_col]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

# Load model and scaler
model, scaler = load_and_train_model()

# Streamlit UI
st.title("🔋 Energy Consumption Predictor")
st.write("Enter real (unstandardized) values below:")

input_str = st.text_input("T_Supply, T_Return, T_Outdoor, T_Saturation", "20.0, 19.0, 55.0, 60.0")

if input_str:
    try:
        values = np.array([float(x.strip()) for x in input_str.split(",")])
        if len(values) != 4:
            st.error("❌ Please enter exactly 4 comma-separated values.")
        else:
            values_scaled = scaler.transform(values.reshape(1, -1))
            prediction = model.predict(values_scaled)[0]
            st.success(f"✅ Predicted Energy Consumption: {prediction:.2f}")
    except Exception as e:
        st.error(f"❌ Invalid input: {e}")
