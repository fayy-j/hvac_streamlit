import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
    st.subheader("📌 Feature Importance")

    # Updated features and importance scores (use % for visual)
    features = ['T_Return', 'T_Saturation', 'T_Supply', 'T_Outdoor', 'RH_Supply', 'RH_Return', 'RH_Outdoor', 'SP_Return']
    importance = [51, 33, 12, 3, 1, 1, 0, 0]

    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance (%)': importance
    }).sort_values('Importance (%)', ascending=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))  # Control plot size

    bars = ax.barh(feature_df['Feature'], feature_df['Importance (%)'], color='teal', edgecolor='black')
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title("Feature Importance (Predefined)", fontsize=14, pad=10)

    # Add value labels next to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width}%', va='center', fontsize=10)

    # Style
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.set_xlim(0, max(importance) + 10)  # leave space for labels
    plt.tight_layout()

    st.pyplot(fig)

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
