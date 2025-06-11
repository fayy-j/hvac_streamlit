import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Energy Consumption Dashboard", layout="centered")

# Load data and train model
@st.cache_data
def load_and_train_model():
    CSV_URL = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/hvac_preprocessed.csv"
    df = pd.read_csv(CSV_URL)

    # Drop Timestamp column if exists
    df = df.drop(columns=["Timestamp"], errors="ignore")

    # Features and target
    feature_cols = ['T_Supply', 'T_Return', 'T_Outdoor', 'T_Saturation']
    target_col = "Energy"

    X = df[feature_cols]
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

# Load model
model, scaler = load_and_train_model()

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        h1 {
            color: #1f3b4d;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 16px;
            padding: 0.5em 1em;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #16658a;
        }
        .footer {
            text-align: center;
            font-size: 0.8rem;
            color: grey;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>HVAC Energy Consumption Predictor</h1>", unsafe_allow_html=True)
st.markdown("Use this tool to estimate energy usage based on HVAC parameters.")

# Input form
with st.form("prediction_form"):
    st.subheader("🧾 Enter Input Values")
    col1, col2 = st.columns(2)

    with col1:
        t_supply = st.number_input("T_Supply (°C)", value=20.0, step=0.1)
        t_outdoor = st.number_input("T_Outdoor (°C)", value=55.0, step=0.1)
    with col2:
        t_return = st.number_input("T_Return (°C)", value=19.0, step=0.1)
        t_saturation = st.number_input("T_Saturation (%)", value=60.0, step=0.1)

    submit = st.form_submit_button("Predict")

# Predict and display
if submit:
    input_array = np.array([[t_supply, t_return, t_outdoor, t_saturation]])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    st.success(f"✅ **Predicted Energy Consumption:** {prediction:.2f} kWh")

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
