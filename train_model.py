# train_model.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from river import compose, linear_model, preprocessing, metrics
from src.swap_optimizer import recommend_swap_window

# Add the /src folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fatigue_model import build_lstm_model, build_lstm_with_attention, features
from data_loader import load_data

# --- Load and split data ---
print("📥 Loading data...")
X, y = load_data(sequence_length=10)
print(f"✅ Loaded: {X.shape[0]} samples | {X.shape[2]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Base LSTM ---
print("\n🧠 Training base LSTM model...")
model_lstm = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
model_lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8, verbose=1)

y_pred_probs_lstm = model_lstm.predict(X_test)
y_pred_lstm = (y_pred_probs_lstm > 0.5).astype(int)

print("\n--- Base LSTM Classification Report ---")
print(classification_report(y_test, y_pred_lstm))

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_lstm.save(os.path.join(model_dir, "fatigue_lstm_base.h5"))

# --- Attention LSTM ---
print("\n🧠 Training LSTM with Attention...")
model_attn = build_lstm_with_attention(input_shape=(X.shape[1], X.shape[2]))
model_attn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8, verbose=1)

y_pred_probs_attn = model_attn.predict(X_test)
y_pred_attn = (y_pred_probs_attn > 0.5).astype(int)

print("\n--- Attention LSTM Classification Report ---")
print(classification_report(y_test, y_pred_attn))

model_attn.save(os.path.join(model_dir, "fatigue_lstm_attention.h5"))

# --- Random Forest (last timestep only) ---
print("\n🌲 Training Random Forest on final timestep features...")
X_rf_train = X_train[:, -1, :]
X_rf_test = X_test[:, -1, :]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_rf_train, y_train)

rf_preds = rf_model.predict(X_rf_test)

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_preds))

# --- Ensemble Voting (LSTM + RF) ---
print("\n🧩 Generating ensemble predictions (LSTM + RF)...")
ensemble_pred = ((y_pred_lstm.reshape(-1) + rf_preds) >= 1).astype(int)

print("\n--- Ensemble Classification Report ---")
print(classification_report(y_test, ensemble_pred))

# --- River Online Learning ---
print("\n🌊 Simulating online learning via River...")

from river import linear_model, preprocessing, metrics

# Step-by-step build to avoid None assignments
scaler = preprocessing.StandardScaler()
model = linear_model.LogisticRegression()

# Create a valid pipeline explicitly
from river.compose import Pipeline
river_pipeline = Pipeline(scaler | model)

river_metric = metrics.Accuracy()
predictions_made = 0

for xi, yi in zip(X_rf_test, y_test):
    x_dict = {f: float(xi[i]) for i, f in enumerate(features)}

    y_river_pred = river_pipeline.predict_one(x_dict)
    if y_river_pred is not None:
        river_metric = river_metric.update(yi, y_river_pred)
        predictions_made += 1

    river_pipeline = river_pipeline.learn_one(x_dict, yi)

if predictions_made > 0:
    print(f"\n📈 River Online Learning Accuracy (on {predictions_made} samples): {river_metric.get():.3f}")
else:
    print("⚠️ River model made no predictions.")

# --- Anomaly Detection ---
print("\n🔍 Running anomaly detection on fatigue indicators...")
df = pd.read_csv("data/lap_data.csv")
anomaly_features = df[["heart_rate", "stress_index", "tire_wear"]]
iso = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = iso.fit_predict(anomaly_features)

df.to_csv("data/lap_data_with_anomalies.csv", index=False)
print("✅ Anomaly-labeled data saved to data/lap_data_with_anomalies.csv")

# Load the data again or use the one with anomaly/fatigue labels
lap_df = pd.read_csv("data/lap_data_with_anomalies.csv")
swap_suggestion = recommend_swap_window(lap_df)
print(f"\n🚨 Swap Recommendation:\n{swap_suggestion}")