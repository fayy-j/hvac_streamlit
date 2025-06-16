import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import time
import streamlit.components.v1 as components
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# --- Streamlit Page Config ---
st.set_page_config(page_title="HVAC Energy Dashboard", layout="wide")

# --- Background Bubbles ---
components.html("""
<style>
body { margin:0; overflow:hidden; }
.bubble {
  position: absolute;
  bottom: -100px;
  background: rgba(255, 192, 203, 0.3);
  border-radius: 50%;
  animation: floatup 10s infinite ease-in;
}
@keyframes floatup {
  0% { transform: translateY(0) scale(0.5); opacity: 1; }
  100% { transform: translateY(-1000px) scale(1); opacity: 0; }
}
</style>
<script>
for (let i = 0; i < 20; i++) {
  let b = document.createElement("div");
  b.className = "bubble";
  b.style.width = b.style.height = (Math.random()*40+20)+"px";
  b.style.left = (Math.random()*100)+"%";
  document.body.appendChild(b);
}
</script>
""", height=0)

# --- Title ---
st.markdown("""
    <h1 style='text-align: center; color: #6A5ACD;'>HVAC Energy Dashboard</h1>
    <p style='text-align: center; color: #999;'>Real-time Simulation and Analysis</p>
""", unsafe_allow_html=True)

# --- Load Model and Data ---
@st.cache_data
def load_model_and_data():
    url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/rawhvac.csv"
    df = pd.read_csv(url, delimiter=';')
    df = df.drop(columns=["Timestamp"], errors="ignore")
    feature_cols = ['T_Supply', 'T_Return', 'T_Outdoor', 'T_Saturation']
    X = df[feature_cols]
    y = df["Energy"]
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, df, X, y, y_pred

model, df, X, y, y_pred = load_model_and_data()
if "energy_buffer" not in st.session_state:
    st.session_state.energy_buffer = deque(maxlen=30)

# --- Controls ---
st.sidebar.title("Control Panel")
t_supply = st.sidebar.slider("T_Supply (°C)", 12.0, 31.0, 15.7, 0.1)
t_return = st.sidebar.slider("T_Return (°C)", 12.0, 26.0, 21.6, 0.1)
t_outdoor = st.sidebar.slider("T_Outdoor (°C)", 2.0, 33.0, 16.3, 0.1)
t_saturation = st.sidebar.slider("T_Saturation (°C)", 12.0, 27.0, 13.7, 0.1)

# --- Simulated Live Prediction ---
st.subheader("Simulated Energy Prediction")
if st.button("Start Simulation"):
    for i in range(30):
        input_array = np.array([[t_supply, t_return, t_outdoor, t_saturation]])
        pred = model.predict(input_array)[0]
        st.session_state.energy_buffer.append(pred)
        st.metric(label="Predicted Energy (Live)", value=f"{pred:.2f} kWh")

        energy_df = pd.DataFrame({
            "Index": list(range(len(st.session_state.energy_buffer))),
            "Energy": list(st.session_state.energy_buffer)
        })

        line_chart = (
            alt.Chart(energy_df)
            .mark_line(interpolate="monotone", color="#4f92ff", strokeWidth=3)
            .encode(x="Index", y="Energy")
            .properties(width=700, height=300)
        )
        st.altair_chart(line_chart, use_container_width=True)
        time.sleep(0.5)

# --- Feature Importance ---
st.subheader("Feature Importance")
features = ['T_Return', 'T_Saturation', 'T_Supply', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor', 'SP_Return']
importance = [51, 33, 12, 3, 1, 1, 0, 0]
feature_df = pd.DataFrame({'Feature': features, 'Importance (%)': importance}).sort_values("Importance (%)")
fig = px.bar(
    feature_df, x="Importance (%)", y="Feature", orientation="h",
    color="Importance (%)", text="Importance (%)",
    color_continuous_scale="bluered_r"
)
fig.update_layout(height=400, plot_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# --- Daily and Hourly ---
st.subheader("Average Energy by Day and Hour")
daily_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/average_energy_by_day.csv"
hourly_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/average_energy_by_hour.csv"

try:
    daily_df = pd.read_csv(daily_url)
    daily_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_df['Day'] = pd.Categorical(daily_df['Day'], categories=daily_order, ordered=True)
    daily_df = daily_df.sort_values("Day")

    hourly_df = pd.read_csv(hourly_url).sort_values("Hour")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Daily Average")
        st.bar_chart(daily_df.set_index("Day"))
    with col2:
        st.markdown("#### Hourly Average")
        st.line_chart(hourly_df.set_index("Hour"))
except Exception as e:
    st.error(f"Failed to load averages: {e}")

# --- Trend ---
st.subheader("Energy Trend Over Time")
try:
    df_trend = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/predicted_vs_actual_energy.csv")
    df_trend['Timestamp'] = pd.to_datetime(df_trend['Timestamp'])
    df_trend = df_trend[['Timestamp', 'Energy']].set_index('Timestamp')
    st.line_chart(df_trend)
except Exception as e:
    st.error(f"Trend data error: {e}")

# --- Actual vs Predicted ---
st.subheader("Actual vs Predicted")
comparison_df = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/predicted_vs_actual_energy.csv")
comparison_df['Index'] = comparison_df.index
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=comparison_df['Index'], y=comparison_df['Actual_Energy'], name='Actual', line=dict(color='skyblue')))
fig2.add_trace(go.Scatter(x=comparison_df['Index'], y=comparison_df['Predicted_Energy'], name='Predicted', line=dict(color='pink')))
fig2.update_layout(title="Actual vs Predicted", xaxis_title="Index", yaxis_title="Energy (kWh)", height=500)
st.plotly_chart(fig2, use_container_width=True)

# --- Footer ---
st.markdown("""
---
<div style='text-align:center;font-size:12px;color:gray'>
    © 2025 fayy-j | HVAC Dashboard 
</div>
""", unsafe_allow_html=True)
