import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# === Load scaler.pkl locally ===
scaler = joblib.load("scaler.pkl")

# === Define model.pkl Google Drive direct link ===
# Replace with your actual file ID from Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1X0166oZTUgdAsGl-H5q93gxuJOXwbEZH"

# Function to load model.pkl from Google Drive
@st.cache_resource
def load_model_from_drive(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        st.error("Failed to load model from Google Drive.")
        return None

model = load_model_from_drive(MODEL_URL)

# === Streamlit App UI ===
st.title("Energy Consumption Prediction")
st.markdown("Enter standardized values to predict energy consumption:")

# Input features
feature_names = ['T_Supply', 'T_Return', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor']

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
    except Exception as e:
        st.error(f"Invalid input: {e}")
