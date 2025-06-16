import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add the /src folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fatigue_model import build_lstm_model, features
from data_loader import load_data

# --- Load and split data ---
print("📥 Loading data...")
X, y = load_data(sequence_length=10)
print(f"✅ Loaded: {X.shape[0]} samples | {X.shape[2]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Build model ---
print("🧠 Building model...")
model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

# --- Train model ---
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=8,
    verbose=1
)

# --- Evaluate model ---
print("📊 Evaluating...")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# --- Optional: Save model weights ---
os.makedirs("model", exist_ok=True)
model.save("model/fatigue_lstm.h5")
print("💾 Model saved to model/fatigue_lstm.h5")