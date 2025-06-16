import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from src.fatigue_model import features
from src.swap_optimizer import recommend_swap_window

st.set_page_config(page_title="Driver Fatigue & Swap Optimizer", layout="wide")
st.title("🏁 Driver Fatigue & Swap Optimizer")

# --- File Upload ---
st.sidebar.header("📂 Upload Lap Data")
uploaded_file = st.sidebar.file_uploader("Upload your .csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} laps")

    # --- Show Raw Data ---
    with st.expander("🔍 Preview Uploaded Data"):
        st.dataframe(df.head(), use_container_width=True)

    # --- Fatigue Prediction ---
    st.header("🧠 Fatigue Prediction")

    try:
        model = load_model("model/fatigue_lstm_attention.h5")
    except:
        st.error("Model file not found! Ensure 'fatigue_lstm_with_attention.h5' is in /model.")
        st.stop()

    # Create sequences of 10 laps
    SEQ_LEN = 10
    sequences = []
    for i in range(len(df) - SEQ_LEN):
        window = df[features].iloc[i:i+SEQ_LEN].values
        sequences.append(window)
    X_seq = np.array(sequences)

    y_pred_probs = model.predict(X_seq)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    df['fatigued'] = 0
    df.loc[SEQ_LEN:, 'fatigued'] = y_pred  # Offset for prediction start

    # Save labeled data for reuse
    df.to_csv("data/lap_data_with_anomalies.csv", index=False)

    # --- Chart: Fatigue over Time ---
    st.header("📈 Fatigue Timeline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['lap'], y=df['lap_time'], mode='lines+markers', name='Lap Time'))
    fig.add_trace(go.Scatter(x=df['lap'], y=df['fatigued'] * df['lap_time'].max(), mode='markers',
                             name='Fatigued', marker=dict(size=10, color='red')))
    st.plotly_chart(fig, use_container_width=True)

    # --- Swap Recommendation ---
    st.header("🔄 Swap Recommendation")
    swap_suggestion = recommend_swap_window(df)
    st.markdown(f"**🔁 Suggested Swap Window:** {swap_suggestion}")

    with st.expander("🧪 View Detected Fatigue Laps"):
        st.dataframe(df[df['fatigued'] == 1], use_container_width=True)

else:
    st.warning("Please upload lap data to begin.")