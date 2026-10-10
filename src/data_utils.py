"""Generate and validate synthetic network-flow data."""

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "duration", "src_bytes", "dst_bytes",
    "failed_logins", "connection_count",
    "unique_destinations", "packet_rate",
    "bytes_per_second", "protocol_code", "service_code"
]

TARGET = "label"


def generate_synthetic_data(n_samples=2500, random_state=42):
    rng = np.random.default_rng(random_state)
    frames = []

    for label, count in [
        ("normal", n_samples // 2),
        ("attack", n_samples - n_samples // 2)
    ]:
        attack = label == "attack"

        duration = rng.gamma(
            0.8 if attack else 2.0,
            1.2 if attack else 4.0,
            count
        )
        src_bytes = rng.lognormal(
            5 if attack else 7, 1, count
        ).astype(int)
        dst_bytes = rng.lognormal(
            4.3 if attack else 7.2, 1.1, count
        ).astype(int)

        frame = pd.DataFrame({
            "duration": duration.round(3),
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "failed_logins": rng.poisson(
                2.4 if attack else 0.08, count
            ),
            "connection_count": rng.poisson(
                28 if attack else 5, count
            ),
            "unique_destinations": rng.poisson(
                10 if attack else 2, count
            ),
            "packet_rate": rng.gamma(
                7 if attack else 3,
                18 if attack else 5,
                count
            ).round(3),
            "bytes_per_second": (
                (src_bytes + dst_bytes) /
                np.maximum(duration, 0.1)
            ).round(3),
            "protocol_code": rng.integers(0, 3, count),
            "service_code": rng.integers(0, 8, count),
            "label": label
        })

        frames.append(frame)

    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def load_dataset(path):
    df = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    df[TARGET] = (
        df[TARGET].astype(str).str.strip().str.lower()
        .replace({
            "benign": "normal",
            "malicious": "attack"
        })
    )

    if not set(df[TARGET].unique()).issubset(
        {"normal", "attack"}
    ):
        raise ValueError(
            "Labels must be normal/attack or benign/malicious."
        )

    if df.empty:
        raise ValueError("Dataset contains no valid rows.")

    return df


def ensure_demo_dataset(path="data/raw/synthetic_network_data.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        generate_synthetic_data().to_csv(path, index=False)

    return path