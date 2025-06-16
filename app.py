import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# --- Streamlit Page Config ---
st.set_page_config(page_title="HVAC Energy Dashboard", layout="wide")

# --- Custom Theme & Style ---
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #f9fafd;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stTabs [role="tablist"] {
            border-bottom: 2px solid #e0e0e0;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 4px solid #a29bfe;
            color: #6c5ce7;
        }
        .metric-box {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .footer {
            text-align: center;
            font-size: 13px;
            color: gray;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align:center; color:#6c5ce7;'>HVAC Energy Consumption Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Simulate, analyze, and visualize HVAC energy usage with style.</p>", unsafe_allow_html=True)

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

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Predict Energy", "Feature Importance", "Daily & Hourly Averages", "Consumption Trend", "Actual vs Predicted",
])

# --- Tab 1 ---
with tab1:
    st.markdown("<h3 style='color:#6c5ce7;'>What-If Simulation</h3>", unsafe_allow_html=True)

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

    st.markdown(f"""
        <div class="metric-box">
            <h2 style="margin: 0;">{prediction:.2f} kWh</h2>
            <p style="margin: 0; color:#555;">Predicted Energy</p>
        </div>
    """, unsafe_allow_html=True)

    energy_df = pd.DataFrame({
        "Index": list(range(len(st.session_state.energy_buffer))),
        "Energy": list(st.session_state.energy_buffer)
    })

    line_chart = (
        alt.Chart(energy_df)
        .mark_line(interpolate="monotone", color="#f78fb3", strokeWidth=3)
        .encode(
            x=alt.X("Index", title="Time (simulated updates)"),
            y=alt.Y("Energy", title="Predicted Energy (kWh)")
        )
        .properties(width=600, height=300, title="Predicted Energy Trend")
    )
    st.altair_chart(line_chart, use_container_width=True)

# --- Tab 2 ---
with tab2:
    st.markdown("<h3 style='color:#6c5ce7;'>Feature Importance</h3>", unsafe_allow_html=True)

    features = ['T_Return', 'T_Saturation', 'T_Supply', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor', 'SP_Return']
    importance = [51, 33, 12, 3, 1, 1, 0, 0]

    feature_df = pd.DataFrame({'Feature': features, 'Importance (%)': importance}).sort_values('Importance (%)')

    fig = px.bar(
        feature_df, x='Importance (%)', y='Feature', orientation='h', text='Importance (%)',
        color='Importance (%)', color_continuous_scale='RdPu', height=400
    )
    fig.update_layout(title="Feature Importance", plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3 ---
with tab3:
    st.markdown("<h3 style='color:#6c5ce7;'>Daily and Hourly Averages</h3>", unsafe_allow_html=True)

    daily_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/average_energy_by_day.csv"
    hourly_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/average_energy_by_hour.csv"

    try:
        daily_df = pd.read_csv(daily_url).sort_values("Day").set_index("Day")
        hourly_df = pd.read_csv(hourly_url).sort_values("Hour").set_index("Hour")

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(daily_df["Energy"])
        with col2:
            st.line_chart(hourly_df["Energy"])
    except Exception as e:
        st.error(f"Failed to load CSVs: {e}")

# --- Tab 4 ---
with tab4:
    st.markdown("<h3 style='color:#6c5ce7;'>Energy Consumption Trend</h3>", unsafe_allow_html=True)

    try:
        df_trend = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/predicted_vs_actual_energy.csv")
        df_trend['Timestamp'] = pd.to_datetime(df_trend['Timestamp'])
        df_trend = df_trend[['Timestamp', 'Energy']].set_index('Timestamp')
        st.line_chart(df_trend)
    except Exception as e:
        st.error(f"Failed to load trend data: {e}")

# --- Tab 5 ---
with tab5:
    st.markdown("<h3 style='color:#6c5ce7;'>Actual vs Predicted Energy</h3>", unsafe_allow_html=True)

    comparison_df = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/predicted_vs_actual_energy.csv")
    comparison_df['Index'] = comparison_df.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison_df['Index'], y=comparison_df['Actual_Energy'],
                             mode='lines', name='Actual Energy (kWh)', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=comparison_df['Index'], y=comparison_df['Predicted_Energy'],
                             mode='lines', name='Predicted Energy (kWh)', line=dict(color='orange')))

    fig.update_layout(title="Actual vs Predicted Energy (Zoomable)",
                      xaxis_title="Sample Index", yaxis_title="Energy (kWh)",
                      hovermode="x unified", height=500,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("""
    <div class="footer">
        © 2025 fayy-j · HVAC Energy Dashboard
    </div>
""", unsafe_allow_html=True)
