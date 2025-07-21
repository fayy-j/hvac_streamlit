import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import deque
import altair as alt

# --- Page Config ---
st.set_page_config(page_title="HVAC Energy Dashboard", layout="wide")

# --- Title ---
st.markdown("""
    <h1 style='text-align: center; color:#333;'>
        HVAC Energy Consumption Dashboard
    </h1>
    <p style='text-align: center; font-size:18px;'>
        Analyze, simulate, and monitor energy patterns.
    </p>
""", unsafe_allow_html=True)

# --- Optional Lottie Animation ---
try:
    from streamlit_lottie import st_lottie
    import requests
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/Animation%20-%201750105697134.json"
    lottie_json = load_lottieurl(lottie_url)

    if lottie_json:
        st_lottie(lottie_json, height=80, speed=1, key="animation", quality="low", loop=True)
except:
    st.markdown("<div style='text-align:center; font-size:30px;'>🚶‍♀️</div>", unsafe_allow_html=True)

# --- Load Model and Data ---
@st.cache_data

def load_model_and_data():
    data_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/rawhvac.csv"
    df = pd.read_csv(data_url, delimiter=';')
    df = df.drop(columns=["Timestamp"], errors="ignore")

    target = 'Energy'
    selected_features = ['T_Return', 'T_Saturation', 'T_Supply', 'T_Outdoor', 'RH_Return', 'RH_Supply']
    X = df[selected_features]
    y = df[target]

    # Split 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train with 70:30 parameter config
    model = RandomForestRegressor(n_estimators=400, min_samples_split=2, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, df, mae, rmse, r2

model, df, mae, rmse, r2 = load_model_and_data()

if "energy_buffer" not in st.session_state:
    st.session_state.energy_buffer = deque(maxlen=30)

# --- Section: Predict Energy ---
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

    line_chart = (
        alt.Chart(energy_df)
        .mark_line(interpolate="monotone", color="orange", strokeWidth=3)
        .encode(
            x=alt.X("Index", title="Time (simulated updates)"),
            y=alt.Y("Energy", title="Predicted Energy (kWh)")
        )
        .properties(width=600, height=300, title="Predicted Energy Trend")
    )

    st.altair_chart(line_chart, use_container_width=True)

# --- Notice: Model Performance ---
st.info(f"""
⚠️ **Notice on Prediction Accuracy**

The prediction model explains approximately **{r2:.2%}** of the variation in energy consumption (R² score).

Expected prediction error:
- **MAE** ≈ {mae:.2f} kWh (average absolute error)
- **RMSE** ≈ {rmse:.2f} kWh (root mean squared error)

Use predictions with awareness of this potential margin of error.
""")

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 13px; color: gray;'>
        © 2025 fayy-j · HVAC Energy Consumption Estimator
    </div>
""", unsafe_allow_html=True)
