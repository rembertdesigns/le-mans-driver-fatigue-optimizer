from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# 👇 Global feature list used in LSTM input pipeline
features = [
    "lap_time", "avg_speed", "heart_rate", "stress_index", "tire_wear",
    "fuel_load", "track_temp", "weather_factor", "sector_1", "sector_2", "sector_3",
    "avg_g_force", "steering_input_intensity", "prior_stint_avg_laptime",
    "prior_stint_duration", "local_hour", "circadian_dip"
]

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))  # output: fatigue score (0 to 1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model