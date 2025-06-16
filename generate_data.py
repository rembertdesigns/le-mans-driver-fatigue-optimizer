import pandas as pd
import numpy as np

# Configuration
num_laps = 100  # adjust as needed
base_lap_time = 220  # seconds
base_speed = 200  # km/h
base_hr = 120  # bpm
base_stress = 0.3

np.random.seed(42)

laps = []
for lap in range(1, num_laps + 1):
    time = (lap - 1) * base_lap_time

    # Fatigue sim: HR & stress go up, speed & performance go down slightly over time
    fatigue_factor = lap / num_laps

    lap_time = base_lap_time + np.random.normal(0, 1.5) + fatigue_factor * 5
    avg_speed = base_speed - fatigue_factor * 8 + np.random.normal(0, 1.2)
    heart_rate = base_hr + fatigue_factor * 25 + np.random.normal(0, 3)
    stress_index = base_stress + fatigue_factor * 0.4 + np.random.normal(0, 0.05)

    laps.append({
        "lap": lap,
        "time": round(time, 2),
        "lap_time": round(lap_time, 2),
        "avg_speed": round(avg_speed, 2),
        "heart_rate": int(heart_rate),
        "stress_index": round(min(stress_index, 1.0), 2)  # max cap at 1.0
    })

df = pd.DataFrame(laps)
df.to_csv("data/lap_data.csv", index=False)
print("✅ Simulated lap_data.csv created in /data")
