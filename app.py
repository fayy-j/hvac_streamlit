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

# --- Cute Theme with Emojis and Pastel Vibes ---
st.markdown("""
    <style>
    .main {
        background-color: #fff9f9;
        color: #4f4f4f;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #fff3f8;
        border: 1px solid #ffb6c1;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stButton button {
        background-color: #ffccd5;
        color: black;
        border-radius: 12px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #ffaebd;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title with emoji
st.title("💖 Energy Predictor Dashboard")
st.subheader("✨ A gentle & cute tool to predict HVAC energy usage ✨")

st.markdown("Enter the following values to get your cozy energy estimate:")

input_str = st.text_input("🌡️ T_Supply, T_Return, T_Outdoor, T_Saturation", "20.0, 19.0, 55.0, 60.0")

if input_str:
    try:
        values = np.array([float(x.strip()) for x in input_str.split(",")])
        if len(values) != 4:
            st.error("🚫 Oopsie! Please enter **exactly 4** values 💭")
        else:
            values_scaled = scaler.transform(values.reshape(1, -1))
            prediction = model.predict(values_scaled)[0]
            st.success(f"🌟 Your predicted energy consumption is: **{prediction:.2f} kWh** 💡")
    except Exception as e:
        st.error(f"😿 Uh-oh! Something went wrong: {e}")
