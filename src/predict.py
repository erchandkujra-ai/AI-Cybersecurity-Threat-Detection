"""Predict suspicious activity using a trained model."""

from pathlib import Path
import joblib
import pandas as pd

from src.data_utils import FEATURES


def load_model(model_path="models/threat_detector.joblib"):
    path = Path(model_path)

    if not path.exists():
        from src.train import train_model
        train_model(model_path=str(path))

    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_dataframe(
    df,
    model_path="models/threat_detector.joblib"
):
    model, features = load_model(model_path)

    missing = [c for c in features if c not in df.columns]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[features].copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        raise ValueError("Feature columns contain invalid values.")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    attack_probability = (
        probabilities[:, classes.index("attack")]
        if "attack" in classes
        else [0.0] * len(df)
    )

    result = df.copy()
    result["prediction"] = predictions
    result["attack_probability"] = attack_probability

    result["severity"] = [
        "High" if p >= 0.85 else
        "Medium" if p >= 0.60 else
        "Low"
        for p in attack_probability
    ]

    return result