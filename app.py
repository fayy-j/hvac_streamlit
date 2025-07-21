import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Title ---
st.set_page_config(page_title="HVAC Model Evaluation", layout="centered")
st.title("🏢 HVAC Energy Prediction Model Evaluation")

# --- Google Drive model link ---
file_id = "1-Sf0WimFvzJjxCeyejJ5g_4seTrR-_Su"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"
model_filename = "rf_hvac_model_8020_compressed.joblib"

# --- Download model if not already exists ---
@st.cache_data(show_spinner=False)
def download_model(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

with st.spinner("📥 Downloading model from Google Drive..."):
    download_model(gdrive_url, model_filename)
    st.success("✅ Model downloaded and cached!")

# --- Load model ---
try:
    model = joblib.load(model_filename)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Upload test files ---
st.subheader("📂 Upload Test Data")
x_file = st.file_uploader("Upload `X_test.csv`", type=["csv"])
y_file = st.file_uploader("Upload `y_test.csv`", type=["csv"])

# --- Process once both files uploaded ---
if x_file is not None and y_file is not None:
    X_test = pd.read_csv(x_file)
    y_test = pd.read_csv(y_file)

    # --- Predict ---
    y_pred = model.predict(X_test)

    # --- Metrics ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # --- Results ---
    st.subheader("📊 Model Performance")
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("MAE (kWh)", f"{mae:.4f}")
    st.metric("RMSE (kWh)", f"{rmse:.4f}")

    st.markdown("---")
    st.info(f"""
    ✅ **Interpretation**:
    - R² of **{r2:.2f}** indicates the model explains **{r2*100:.1f}%** of the variation.
    - MAE of **{mae:.2f} kWh** means average prediction error is small.
    - RMSE of **{rmse:.2f} kWh** reflects how concentrated prediction errors are.

    📎 You can now compare these metrics with your cross-validation results.
    """)
else:
    st.warning("Please upload both `X_test.csv` and `y_test.csv` files to proceed.")
