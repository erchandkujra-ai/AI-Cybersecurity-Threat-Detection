"""Generate synthetic network activity for testing."""

import argparse
from pathlib import Path
from src.data_utils import generate_synthetic_data


def simulate(output, rows=2500, seed=42):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data(rows, seed)
    df.to_csv(path, index=False)

    print(f"Generated {len(df)} records: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/raw/synthetic_network_data.csv"
    )
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    simulate(args.output, args.rows, args.seed)