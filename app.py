import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# --- Streamlit Page Config ---
st.set_page_config(page_title="HVAC Energy Dashboard", layout="wide")

# --- Header ---
st.markdown("<h1 style='text-align:center;'>HVAC Energy Consumption Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Get predictions, explore feature importance, and visualize consumption patterns.")

# --- Load Model and Data ---
@st.cache_data
def load_model_and_data():
    # Load data with semicolon delimiter
    data_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/rawhvac.csv"
    df = pd.read_csv(data_url, delimiter=';')
    
    # Drop timestamp if exists
    df = df.drop(columns=["Timestamp"], errors="ignore")

    # Define features and target
    feature_cols = ['T_Supply', 'T_Return', 'T_Outdoor', 'T_Saturation']
    X = df[feature_cols]
    y = df["Energy"]

    # Use optimized Random Forest without scaling
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)

    return model, df, X, y, y_pred

# Load everything
model, df, X, y, y_pred = load_model_and_data()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predict Energy", 
    "📊 Feature Importance", 
    "📆 Daily & Hourly Averages", 
    "📈 Consumption Trend", 
    "🎯 Actual vs Predicted",
    "🧪 Dynamic What-If"
])

# --- Tab 1: Predict Energy ---
with tab1:
    st.subheader("🔧 What-If Simulation: Predict Energy")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            t_supply = st.number_input("T_Supply (°C)", value=20.0, step=0.1)
            t_outdoor = st.number_input("T_Outdoor (°C)", value=55.0, step=0.1)
        with col2:
            t_return = st.number_input("T_Return (°C)", value=19.0, step=0.1)
            t_saturation = st.number_input("T_Saturation (%)", value=60.0, step=0.1)

        submit = st.form_submit_button("Predict")

    if submit:
        input_array = np.array([[t_supply, t_return, t_outdoor, t_saturation]])
        prediction = model.predict(input_array)[0]
        st.success(f"⚡ Predicted Energy Consumption: **{prediction:.2f} kWh**")

# --- Tab 2: Feature Importance ---

with tab2:
    st.subheader("📌 Feature Importance (Interactive)")

    # Updated features and importance scores
    features = ['T_Return', 'T_Saturation', 'T_Supply', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor', 'SP_Return']
    importance = [51, 33, 12, 3, 1, 1, 0, 0]

    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance (%)': importance
    }).sort_values('Importance (%)', ascending=True)

    # Create interactive horizontal bar chart
    fig = px.bar(
        feature_df,
        x='Importance (%)',
        y='Feature',
        orientation='h',
        text='Importance (%)',
        color='Importance (%)',
        color_continuous_scale='Tealgrn',
        height=400  # make it smaller
    )

    fig.update_traces(
        textposition='outside',
        marker_line_color='black',
        marker_line_width=0.8
    )

    fig.update_layout(
        title="Feature Importance (Predefined)",
        xaxis_title="Importance (%)",
        yaxis_title=None,
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Daily and Hourly Averages ---
with tab3:
    st.subheader("📅 Daily and Hourly Energy Averages")

    daily_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/avg_energy_by_day.csv"
    hourly_url = "https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/hourly_avg_energy.csv"

    try:
        # Load daily average data
        daily_df = pd.read_csv(daily_url)
        daily_df = daily_df.sort_values('DayOfWeek')  # Optional: ensure day order
        daily_df = daily_df.rename(columns={"DayOfWeek": "index"}).set_index("index")

        # Load hourly average data
        hourly_df = pd.read_csv(hourly_url)
        hourly_df = hourly_df.sort_values('Hour')
        hourly_df = hourly_df.rename(columns={"Hour": "index"}).set_index("index")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📅 Average Energy by Day")
            st.bar_chart(daily_df)

        with col2:
            st.markdown("### 🕒 Average Energy by Hour")
            st.line_chart(hourly_df)

    except Exception as e:
        st.error(f"❌ Failed to load or plot CSVs: {e}")


# --- Tab 4: Consumption Trend ---
with tab4:
    st.subheader("📈 Energy Consumption Trend")

    try:
        df_trend = pd.read_csv("https://raw.githubusercontent.com/fayy-j/hvac_streamlit/refs/heads/main/rawhvac.csv")
        df_trend['Timestamp'] = pd.to_datetime(df_trend['Timestamp'])
        df_trend = df_trend[['Timestamp', 'Energy']].set_index('Timestamp')

        st.line_chart(df_trend)

    except Exception as e:
        st.error(f"Failed to load trend data: {e}")

# --- Tab 5: Actual vs Predicted ---
with tab5:
    st.subheader("🎯 Actual vs Predicted Energy (Full Dataset)")

    comparison_df = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred
    })

    st.line_chart(comparison_df)
