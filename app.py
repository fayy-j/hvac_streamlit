import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import altair as alt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import deque

# --- Page Config ---
st.set_page_config(page_title="HVAC Energy Dashboard", layout="wide")

# --- Title ---
st.markdown("""
    <h1 style='text-align: center; color:#333;'>HVAC Energy Consumption Dashboard</h1>
    <p style='text-align: center; font-size:18px;'>Analyze, simulate, and monitor energy patterns.</p>
""", unsafe_allow_html=True)

# --- Lottie Animation ---
try:
    from streamlit_lottie import st_lottie

    def load_lottieurl(url):
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None

    lottie_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/Animation%20-%201750105697134.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=80, speed=1, key="animation", quality="low", loop=True)
except:
    st.markdown("<div style='text-align:center; font-size:30px;'>🚶‍♀️</div>", unsafe_allow_html=True)

# --- Load model and data from Google Drive ---
@st.cache_data
def download_from_drive():
    model_url = "https://drive.google.com/uc?id=1-Sf0WimFvzJjxCeyejJ5g_4seTrR-_Su"
    x_test_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/main/X_test.csv"
    y_test_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/main/y_test.csv"

    model_filename = "rf_hvac_model_8020_compressed.joblib"
    if not os.path.exists(model_filename):
        r = requests.get(model_url)
        with open(model_filename, "wb") as f:
            f.write(r.content)

    model = joblib.load(model_filename)
    X_test = pd.read_csv(x_test_url)
    y_test = pd.read_csv(y_test_url).squeeze()

    return model, X_test, y_test

model, X_test, y_test = download_from_drive()

# --- Evaluation ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

if "energy_buffer" not in st.session_state:
    st.session_state.energy_buffer = deque(maxlen=30)

# --- What-If Simulation ---
with st.expander("🔧 What-If Simulation: Predict Energy", expanded=True):
    st.markdown("Adjust parameters to observe predicted energy in real-time.")

    col1, col2 = st.columns(2)
    with col1:
        t_return = st.slider("T_Return (°C)", 12.0, 26.0, 21.6, 0.1)
        t_saturation = st.slider("T_Saturation (°C)", 12.0, 27.0, 13.7, 0.1)
        t_supply = st.slider("T_Supply (°C)", 12.0, 31.0, 15.7, 0.1)
    with col2:
        t_outdoor = st.slider("T_Outdoor (°C)", 2.0, 33.0, 16.3, 0.1)
        rh_return = st.slider("RH_Return (%)", 11.0, 80.0, 45.0, 0.1)
        rh_supply = st.slider("RH_Supply (%)", 19.0, 85.0, 50.0, 0.1)

    input_array = np.array([[t_return, t_saturation, t_supply, t_outdoor, rh_return, rh_supply]])
    prediction = model.predict(input_array)[0]
    st.session_state.energy_buffer.append(prediction)

    st.metric(label="Predicted Energy", value=f"{prediction:.2f} kWh")

    energy_df = pd.DataFrame({
        "Index": list(range(len(st.session_state.energy_buffer))),
        "Energy": list(st.session_state.energy_buffer)
    })

    chart = (
        alt.Chart(energy_df)
        .mark_line(interpolate="monotone", color="orange", strokeWidth=3)
        .encode(
            x=alt.X("Index", title="Time (simulated updates)"),
            y=alt.Y("Energy", title="Predicted Energy (kWh)")
        )
        .properties(width=600, height=300)
    )
    st.altair_chart(chart, use_container_width=True)

# --- Accuracy Notice ---
st.info(f"""
⚠️ **Notice on Prediction Accuracy**

The prediction model explains approximately **{r2:.2%}** of the variation in energy consumption (R² score).

Expected prediction error:
- **MAE** ≈ {mae:.2f} kWh
- **RMSE** ≈ {rmse:.2f} kWh

Use predictions with awareness of this margin of error.
""")

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 13px; color: gray;'>
        © 2025 fayy-j · HVAC Energy Consumption Estimator
    </div>
""", unsafe_allow_html=True)
