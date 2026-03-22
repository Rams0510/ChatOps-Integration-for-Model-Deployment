"""
Phase 1 — Step 1: Train and save the ML model.
Run this ONCE locally before building Docker:
    python train_model.py
"""
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train():
    print("Loading dataset...")
    X, y = load_iris(return_X_y=True)
    target_names = ["setosa", "versicolor", "virginica"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print(f"\nTest accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=target_names))

    if accuracy < 0.90:
        raise ValueError(
            f"Model accuracy {accuracy:.3f} is below threshold 0.90. Not saving."
        )

    joblib.dump(model, "app/model.pkl")
    print("Model saved to app/model.pkl")


if __name__ == "__main__":
    train()
