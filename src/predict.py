import pandas as pd
import joblib

# Load model and feature names used during training
model = joblib.load("models/energy_model.pkl")
expected_features = joblib.load("models/feature_names.pkl")

def predict(temp, humidity, hour, day_of_week, appliance):
    # Create DataFrame with input
    data = {
        "temperature": [temp],
        "humidity": [humidity],
        "hour": [hour],
        "day_of_week": [day_of_week],
    }
    
    # Add one-hot encoded appliances
    for app in ["motor", "fan", "ac", "heater"]:
        data[f"appliance_{app}"] = [1 if appliance == app else 0]

    df = pd.DataFrame(data)

    # 🔑 Align columns with training
    df = df.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(df)[0]
    return prediction


if __name__ == "__main__":
    print("🔮 Predicted energy usage:", predict(30, 50, 14, 2, "motor"))
