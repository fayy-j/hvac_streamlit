import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input feature names
feature_names = ['T_Supply', 'T_Return', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor']

st.title("Energy Consumption Prediction")
st.markdown("Enter the input values to predict energy consumption.")

# Collect inputs from user all at once
user_input = st.text_input(
    label="Enter standardized values for: T_Supply, T_Return, T_Outdoor, RH_Supply, RH_Return, RH_Outdoor",
    placeholder="Example: -0.045, 0.25, 1.55, 2.74, 2.78, 1.27"
)

if user_input:
    try:
        values = np.array([float(x.strip()) for x in user_input.split(',')])
        if len(values) != 6:
            st.error("Please enter exactly 6 values.")
        else:
            prediction = model.predict(values.reshape(1, -1))[0]
            st.success(f"Predicted Energy Consumption: {prediction:.4f}")
    except:
        st.error("Invalid input. Please enter comma-separated numeric values.")
