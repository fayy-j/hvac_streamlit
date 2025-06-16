import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
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

# --- Load Model and Data ---
@st.cache_data

def load_model_and_data():
    data_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/rawhvac.csv"
    df = pd.read_csv(data_url, delimiter=';')
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

# --- Section 1: Predict Energy ---
with st.expander("🔧 What-If Simulation: Predict Energy", expanded=True):
    st.markdown("Adjust parameters to observe predicted energy in real-time.")

    col1, col2 = st.columns(2)
    with col1:
        t_supply = st.slider("T_Supply (°C)", 12.0, 31.0, 15.7, 0.1)
        t_outdoor = st.slider("T_Outdoor (°C)", 2.0, 33.0, 16.3, 0.1)
    with col2:
        t_return = st.slider("T_Return (°C)", 12.0, 26.0, 21.6, 0.1)
        t_saturation = st.slider("T_Saturation (°C)", 12.0, 27.0, 13.7, 0.1)

    input_array = np.array([[t_supply, t_return, t_outdoor, t_saturation]])
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


# --- Section 2: Feature Importance ---
with st.expander("📌 Feature Importance", expanded=False):
    features = ['T_Return', 'T_Saturation', 'T_Supply', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor', 'SP_Return']
    importance = [51, 33, 12, 3, 1, 1, 0, 0]

    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance (%)': importance
    }).sort_values('Importance (%)', ascending=True)

    fig = px.bar(
        feature_df,
        x='Importance (%)',
        y='Feature',
        orientation='h',
        text='Importance (%)',
        color='Importance (%)',
        color_continuous_scale='Tealgrn',
        height=400
    )
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance (%)",
        yaxis_title=None,
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Section 3: Daily and Hourly Averages ---
with st.expander("📅 Daily & Hourly Energy Averages", expanded=False):
    try:
        daily_df = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/average_energy_by_day.csv")
        hourly_df = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/average_energy_by_hour.csv")

        daily_df = daily_df.sort_values("Day").set_index("Day")
        hourly_df = hourly_df.sort_values("Hour").set_index("Hour")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Average Energy by Day")
            st.bar_chart(daily_df["Energy"])
        with col2:
            st.markdown("### Average Energy by Hour")
            st.line_chart(hourly_df["Energy"])

    except Exception as e:
        st.error(f"Error loading averages: {e}")


# --- Section 4: Consumption Trend ---
with st.expander("📈 Historical Energy Consumption Trend", expanded=False):
    try:
        df_trend = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/predicted_vs_actual_energy.csv")
        df_trend['Timestamp'] = pd.to_datetime(df_trend['Timestamp'])
        df_trend = df_trend[['Timestamp', 'Energy']].set_index('Timestamp')
        st.line_chart(df_trend)
    except Exception as e:
        st.error(f"Error loading trend data: {e}")


# --- Section 5: Actual vs Predicted ---
with st.expander("🎯 Actual vs Predicted Energy", expanded=False):
    try:
        df_comp = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/predicted_vs_actual_energy.csv")
        df_comp['Index'] = df_comp.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_comp['Index'], y=df_comp['Actual_Energy'], mode='lines', name='Actual', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=df_comp['Index'], y=df_comp['Predicted_Energy'], mode='lines', name='Predicted', line=dict(color='orange')))

        fig.update_layout(
            title="Actual vs Predicted Energy",
            xaxis_title="Index",
            yaxis_title="Energy (kWh)",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1, yanchor="top")
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading actual vs predicted data: {e}")


# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 13px; color: gray;'>
        © 2025 fayy-j · HVAC Energy Dashboard
    </div>
""", unsafe_allow_html=True)
