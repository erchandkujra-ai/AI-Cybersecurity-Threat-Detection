"""Train and evaluate the Random Forest model."""

from pathlib import Path
import argparse
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from src.data_utils import FEATURES, TARGET, load_dataset, ensure_demo_dataset


def train_model(
    data_path="data/raw/synthetic_network_data.csv",
    model_path="models/threat_detector.joblib"
):
    if not Path(data_path).exists():
        ensure_demo_dataset(data_path)

    df = load_dataset(data_path)
    X = df[FEATURES]
    y = df[TARGET]

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test, predictions, zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["normal", "attack"]
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "features": FEATURES
    }, model_path)

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    (output / "training_metrics.txt").write_text(
        f"Rows: {len(df)}\n"
        f"Test accuracy: {accuracy:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix:\n{matrix}\n",
        encoding="utf-8"
    )

    print(f"Model saved: {model_path}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(report)
    print("Confusion matrix:\n", matrix)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/raw/synthetic_network_data.csv"
    )
    parser.add_argument(
        "--model",
        default="models/threat_detector.joblib"
    )
    args = parser.parse_args()

    train_model(args.data, args.model)