# Dataset Guide

## Purpose

The project uses simplified numeric features to demonstrate a defensive
machine-learning workflow. The default dataset is synthetic, not captured
from real networks.

## Required columns

| Column | Meaning |
|---|---|
| duration | Duration-like value |
| src_bytes | Source byte count |
| dst_bytes | Destination byte count |
| failed_logins | Failed-login count |
| connection_count | Connection count |
| unique_destinations | Number of distinct destinations |
| packet_rate | Packet-rate-like value |
| bytes_per_second | Byte rate |
| protocol_code | Numeric protocol category |
| service_code | Numeric service category |
| label | normal or attack |

Training CSV files must contain every feature and the `label` column.
Labels `benign` and `malicious` are also accepted.

For prediction uploads, all numeric feature columns are required, but
the label column is optional.

## Evaluation

Check precision, recall, F1-score, and the confusion matrix. Accuracy
alone can hide poor detection on imbalanced data.

## Privacy

- Use only authorized datasets.
- Do not commit confidential logs or credentials.
- Check dataset licensing and document preprocessing.
- Synthetic test results do not establish real-world accuracy.