import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path="data/lap_data.csv", sequence_length=10):
    df = pd.read_csv(path)

    feature_cols = [
        "lap_time", "avg_speed", "heart_rate", "stress_index", "tire_wear",
        "fuel_load", "track_temp", "weather_factor", "sector_1", "sector_2", "sector_3",
        "avg_g_force", "steering_input_intensity", "prior_stint_avg_laptime",
        "prior_stint_duration", "local_hour", "circadian_dip"
    ]

    label_col = "fatigued"

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(scaled_features[i:i+sequence_length])
        y.append(df[label_col].iloc[i+sequence_length])

    return np.array(X), np.array(y)