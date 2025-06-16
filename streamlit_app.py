import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🏁 Driver Fatigue & Swap Optimizer")
df = pd.read_csv("data/simulated_lap_data.csv")

st.line_chart(df[['lap', 'lap_time']])
st.line_chart(df[['lap', 'heart_rate']])