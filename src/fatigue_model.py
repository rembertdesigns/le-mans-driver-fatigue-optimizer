from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Permute, Multiply, Lambda
from tensorflow.keras import backend as K
import numpy as np

# 👇 Global feature list used in both LSTM and RF pipelines
features = [
    "lap_time", "avg_speed", "heart_rate", "stress_index", "tire_wear",
    "fuel_load", "track_temp", "weather_factor", "sector_1", "sector_2", "sector_3",
    "avg_g_force", "steering_input_intensity", "prior_stint_avg_laptime",
    "prior_stint_duration", "local_hour", "circadian_dip"
]

# ⚙️ Basic 2-layer LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))  # output: fatigue score (0 to 1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 🧠 LSTM with attention mechanism
def build_lstm_with_attention(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inputs)

    # Attention mechanism
    attention_weights = Dense(1, activation='tanh')(lstm_out)
    attention_weights = Permute([2, 1])(attention_weights)
    attention_weights = Dense(input_shape[0], activation='softmax')(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)

    context_vector = Multiply()([lstm_out, attention_weights])
    context_vector = Lambda(lambda x: K.sum(x, axis=1))(context_vector)

    output = Dense(1, activation='sigmoid')(context_vector)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model