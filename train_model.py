"""
Train and save the House Price Prediction model.
Uses the California Housing dataset — a real-world dataset with 20,640 samples.
Run once: python train_model.py
"""
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

def train():
    print("Loading California Housing dataset (20,640 real samples)...")
    data = fetch_california_housing()
    X, y = data.data, data.target  # y is in $100,000s

    print(f"Dataset: {X.shape[0]} houses, {X.shape[1]} features")
    print(f"Price range: ${y.min()*100:.0f}k — ${y.max()*100:.0f}k")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining Gradient Boosting Regressor...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        ))
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    print(f"\nTest MAE:  ${mae*100:.2f}k")
    print(f"Test R²:   {r2:.3f}")

    if r2 < 0.75:
        raise ValueError(f"R² {r2:.3f} too low — not saving model.")

    # Save model + feature names
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": data.feature_names,
        "target_scale": 100_000,  # multiply predictions by this for USD
    }, "app/model.pkl")
    print("\nModel saved to app/model.pkl")
    print(f"Ready to predict house prices with R²={r2:.3f}")

if __name__ == "__main__":
    train()
