# Interview Preparation

### 1. What does the project do?

It demonstrates a machine-learning pipeline that scores tabular network-like
records, assigns a simple severity level, visualizes results, and exports reports.

### 2. Why Random Forest?

It is a useful baseline for tabular data and can model nonlinear relationships.
It is not guaranteed to be the best model for every dataset.

### 3. What is the target variable?

The target is `label`, with classes `normal` and `attack`.

### 4. Why split the data?

A held-out test set estimates performance on records not used for fitting.
The estimate depends on the dataset and split methodology.

### 5. Which evaluation metrics matter?

Precision, recall, F1-score, confusion matrix, and false-positive rate.

### 6. What are the limitations?

The default data is synthetic, features are simplified, and the application
does not inspect live packets or provide production incident response.

### 7. How can the project be improved?

- Evaluate against a properly licensed public cybersecurity dataset.
- Add time-aware validation.
- Tune decision thresholds.
- Track model versions and dataset provenance.
- Add explainability and drift monitoring.
- Test false positives and false negatives carefully.

### 8. Is this a production SOC tool?

No. It is a learning prototype and should not be the sole basis for
incident response or automated blocking.

### 9. How is severity calculated?

The demo maps attack probability to Low, Medium, or High. These thresholds
are illustrative and are not a validated risk standard.

### 10. How can it be described on GitHub?

A local Streamlit dashboard demonstrating Random Forest classification
of synthetic network-flow-like data, with severity heuristics and
downloadable reports.