# 🧠 Le Mens Driver Fatigue Prediction & Swap Optimization

## 🔍 Overview
This project uses machine learning to predict driver fatigue from lap times and telemetry, recommending optimal swap windows to reduce performance dips. Ideal for endurance racing formats like Le Mans.

## 🚀 ML Techniques
- LSTM-based time series model trained on lap data
- Simulated biometric input (e.g., HR, stress index)
- Optimization logic to recommend swap windows in real-time

## 📁 Structure
- `data/`: Raw and processed telemetry + biometric inputs
- `src/`: Fatigue detection model and optimization logic
- `notebooks/`: Prototyping and EDA
- `streamlit_app.py`: Optional real-time dashboard

## ✅ Goals
- Predict onset of driver fatigue
- Recommend ideal swap lap ranges
- Support real-time HIL (human-in-loop) feedback

## 📦 To Run
```bash
pip install -r requirements.txt
python streamlit_app.py
