import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate datetime range (1000 hours of data)
start = datetime(2025, 1, 1, 0, 0, 0)
datetimes = [start + timedelta(hours=i) for i in range(1000)]

# Generate features
temperature = np.random.randint(20, 35, size=1000)   # in °C
humidity = np.random.randint(30, 70, size=1000)      # in %
appliances = ["fan", "light", "motor"]
appliance = np.random.choice(appliances, size=1000)

# Energy usage (Watts)
energy_usage = []
for i in range(1000):
    base = 50 if appliance[i] == "light" else (200 if appliance[i] == "fan" else 400)
    variation = np.random.randint(-20, 20)
    energy_usage.append(base + variation + 0.5 * temperature[i])

# Create DataFrame
df = pd.DataFrame({
    "datetime": datetimes,
    "temperature": temperature,
    "humidity": humidity,
    "appliance": appliance,
    "energy_usage": energy_usage
})

# Save dataset
df.to_csv("data/energy_data.csv", index=False)
print("✅ Dataset generated and saved at data/energy_data.csv")
