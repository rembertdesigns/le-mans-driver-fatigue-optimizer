import pandas as pd
import numpy as np

# Configuration
num_laps = 100
base_lap_time = 220  # seconds
base_speed = 200     # km/h
base_hr = 120        # bpm
base_stress = 0.3

np.random.seed(42)

laps = []

for lap in range(1, num_laps + 1):
    time = (lap - 1) * base_lap_time
    fatigue_factor = lap / num_laps

    # Performance degradation
    lap_time = base_lap_time + np.random.normal(0, 1.5) + fatigue_factor * 5
    avg_speed = base_speed - fatigue_factor * 8 + np.random.normal(0, 1.2)
    heart_rate = base_hr + fatigue_factor * 25 + np.random.normal(0, 3)
    stress_index = base_stress + fatigue_factor * 0.4 + np.random.normal(0, 0.05)

    # Tire & fuel dynamics
    tire_wear = min(1.0, fatigue_factor + np.random.normal(0, 0.03))
    fuel_load = max(0, 100 - lap)  # linear fuel burn

    # Track & weather conditions
    track_temp = np.random.normal(35 - fatigue_factor * 5, 2)
    weather_factor = np.random.choice([0.0, 0.5, 1.0], p=[0.8, 0.15, 0.05])  # dry, damp, wet

    # Sector splits
    s1 = lap_time * 0.33 + np.random.normal(0, 0.4)
    s2 = lap_time * 0.34 + np.random.normal(0, 0.4)
    s3 = lap_time - s1 - s2

    # G-force & steering load
    avg_g = np.random.normal(3.0 + fatigue_factor, 0.2)
    steering_intensity = min(1.0, 0.5 + fatigue_factor * 0.4 + np.random.normal(0, 0.05))

    # Circadian rhythm
    local_hour = (14 + lap // 4) % 24  # simulates race from 2PM
    circadian_dip = 1 if 2 <= local_hour <= 6 else 0

    # Prior stint history
    prior_stint_avg_laptime = base_lap_time + fatigue_factor * 4
    prior_stint_duration = 60 + int(fatigue_factor * 60)

    # Fatigue label logic
    fatigue_label = int(
        (heart_rate > 140) or
        (stress_index > 0.75) or
        (tire_wear > 0.85) or
        (circadian_dip == 1)
    )

    laps.append({
        "lap": lap,
        "time": round(time, 2),
        "lap_time": round(lap_time, 2),
        "avg_speed": round(avg_speed, 2),
        "heart_rate": int(heart_rate),
        "stress_index": round(min(stress_index, 1.0), 2),
        "tire_wear": round(tire_wear, 2),
        "fuel_load": round(fuel_load, 2),
        "track_temp": round(track_temp, 1),
        "weather_factor": weather_factor,
        "sector_1": round(s1, 2),
        "sector_2": round(s2, 2),
        "sector_3": round(s3, 2),
        "avg_g_force": round(avg_g, 2),
        "steering_input_intensity": round(steering_intensity, 2),
        "prior_stint_avg_laptime": round(prior_stint_avg_laptime, 2),
        "prior_stint_duration": prior_stint_duration,
        "local_hour": local_hour,
        "circadian_dip": circadian_dip,
        "fatigued": fatigue_label
    })

df = pd.DataFrame(laps)
df.to_csv("data/lap_data.csv", index=False)
print("✅ Multi-modal lap_data.csv with fatigue labels created in /data")