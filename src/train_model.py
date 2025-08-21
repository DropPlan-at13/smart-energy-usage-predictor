import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Import your preprocessing function
from src.data_preprocess import load_and_preprocess


def train_model():
    # 1️⃣ Load dataset
    df = load_and_preprocess("data/energy_data.csv")

    # 2️⃣ Define features (X) and target (y)
    X = df.drop("energy_usage", axis=1)
    y = df["energy_usage"]

    # 3️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4️⃣ Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5️⃣ Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("✅ Model trained successfully!")
    print(f"📉 MSE: {mse:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    # 6️⃣ Save model + feature names
    joblib.dump(model, "models/energy_model.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    print("💾 Model and feature names saved in 'models/' directory")


if __name__ == "__main__":
    train_model()
