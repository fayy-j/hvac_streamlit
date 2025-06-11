import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load CSV
df = pd.read_csv("hvac_preprocessed.csv")

# Drop timestamp if exists
df = df.drop(columns=["Timestamp"], errors="ignore")

# Features & target
X = df.drop(columns=["Energy"])
y = df["Energy"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
