# 🧠 Le Mans Driver Fatigue Prediction & Swap Optimization

## 🔍 Overview
This project uses machine learning to predict driver fatigue from lap times and telemetry, recommending optimal swap windows to reduce performance dips. It is ideal for endurance racing formats like Le Mans.

## 🚀 ML Techniques
- LSTM-based time series model trained on lap data
- Simulated biometric input (e.g., heart rate, stress index)
- Optimization logic to recommend swap windows in real time

## 📁 Project Structure
- `data/`: Raw and processed telemetry + biometric inputs
- `src/`: Fatigue detection model and swap optimization logic
- `notebooks/`: Prototyping, exploratory data analysis (EDA), and testing
- `streamlit_app.py`: Optional real-time dashboard for visualization and HIL feedback

## ✅ Project Goals
- Accurately predict the onset of driver fatigue
- Recommend ideal driver swap lap ranges
- Enable real-time HIL (human-in-the-loop) interaction and decision support

## 📦 Getting Started

Install dependencies:

```bash
pip install -r requirements.txt

