import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

CSV_URL = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/hvac_preprocessed.csv"

@st.cache_data
def load_and_train_model():
    df = pd.read_csv(CSV_URL)
    df = df.drop(columns=["Timestamp"], errors="ignore")
    X = df[['T_Supply', 'T_Return', 'T_Outdoor', 'T_Saturation']]
    y = df["Energy"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_and_train_model()

# --- Stylish and Professional CSS Theme ---
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .main {
        background: rgba(173, 216, 230, 0.15); /* Soft blue, transparent */
        border-radius: 12px;
        padding: 2rem;
    }
    html, body, [class*="css"] {
        background-color: rgba(173, 216, 230, 0.2); /* lighter transparent soft blue */
    }
    .stTextInput > div > div > input {
        background-color: #f0faff;
        border: 1px solid #b0cbe8;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stButton button {
        background-color: #4da6ff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #3399ff;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title & Instructions ---
st.title("HVAC Energy Consumption Predictor")
st.subheader("Predict your system's energy usage with real-world input values.")

st.markdown("Enter the following real (for now standardized) values:")

# --- Input Section ---
input_str = st.text_input("T_Supply, T_Return, T_Outdoor, T_Saturation", "20.0, 19.0, 55.0, 60.0")

# --- Prediction Output ---
if input_str:
    try:
        values = np.array([float(x.strip()) for x in input_str.split(",")])
        if len(values) != 4:
            st.error("❌ Please enter exactly 4 comma-separated values.")
        else:
            values_scaled = scaler.transform(values.reshape(1, -1))
            prediction = model.predict(values_scaled)[0]
            st.success(f"🔍 Predicted Energy Consumption: **{prediction:.2f} kWh**")
    except Exception as e:
        st.error(f"⚠️ Invalid input: {e}")

# Expandable info
with st.expander("ℹ️ About This App"):
    st.write("""
        This dashboard uses a machine learning model trained on real HVAC data to predict energy consumption based on:
        - **T_Supply**: Supply air temperature
        - **T_Return**: Return air temperature
        - **T_Outdoor**: Outdoor air temperature
        - **T_Saturation**: Saturation level (%)
        - **Still in development**
    """)

# Footer
st.markdown('<div class="footer">Developed by fayy-j | © 2025</div>', unsafe_allow_html=True)
