import pandas as pd

def load_and_preprocess(path="data/energy_data.csv"):
    df = pd.read_csv(path, parse_dates=["datetime"])

    # Feature engineering
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek

    # One-hot encode appliance type
    df = pd.get_dummies(df, columns=["appliance"], drop_first=True)

    return df

if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())
