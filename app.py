"""Streamlit cybersecurity threat detection dashboard."""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_utils import (
    FEATURES,
    ensure_demo_dataset,
    load_dataset
)
from src.predict import predict_dataframe
from src.train import train_model

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AI Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Powered Cybersecurity Threat Detection")
st.caption("Machine learning demonstration using tabular network data.")

DATA_PATH = ROOT / "data/raw/synthetic_network_data.csv"
MODEL_PATH = ROOT / "models/threat_detector.joblib"

ensure_demo_dataset(DATA_PATH)

with st.sidebar:
    st.header("Control Center")
    st.info(
        "Educational prototype. This app does not monitor live traffic "
        "or block network connections."
    )

    if st.button("Train / Retrain Model"):
        try:
            with st.spinner("Training model..."):
                train_model(str(DATA_PATH), str(MODEL_PATH))
            st.success("Training complete.")
        except Exception as exc:
            st.error(str(exc))

st.subheader("1. Select dataset")

uploaded = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Using uploaded dataset.")
    else:
        df = load_dataset(DATA_PATH)
        st.write("Using generated demo dataset.")

    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Records loaded: {len(df):,}")

except Exception as exc:
    st.error(f"Dataset error: {exc}")
    st.stop()

st.subheader("2. Analyze network records")

if st.button("Run Threat Detection", type="primary"):
    try:
        if not MODEL_PATH.exists():
            with st.spinner("Training initial model..."):
                train_model(str(DATA_PATH), str(MODEL_PATH))

        with st.spinner("Analyzing records..."):
            st.session_state["results"] = predict_dataframe(
                df, str(MODEL_PATH)
            )

    except Exception as exc:
        st.error(f"Detection error: {exc}")

results = st.session_state.get("results")

if results is not None:
    total = len(results)
    attacks = int((results["prediction"] == "attack").sum())
    normal = total - attacks
    high = int((results["severity"] == "High").sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total)
    col2.metric("Predicted Attacks", attacks)
    col3.metric("Predicted Normal", normal)
    col4.metric("High Severity", high)

    st.subheader("3. Detection results")

    selected = st.selectbox(
        "Filter records",
        ["All", "Attack", "Normal", "High Severity"]
    )

    view = results

    if selected == "Attack":
        view = results[results["prediction"] == "attack"]
    elif selected == "Normal":
        view = results[results["prediction"] == "normal"]
    elif selected == "High Severity":
        view = results[results["severity"] == "High"]

    st.dataframe(view, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.write("Prediction distribution")
        st.bar_chart(results["prediction"].value_counts())

    with right:
        st.write("Severity distribution")
        st.bar_chart(results["severity"].value_counts())

    st.subheader("4. Download reports")

    st.download_button(
        "Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="threat_detection_results.csv",
        mime="text/csv"
    )

    summary = (
        "AI Cybersecurity Threat Detection Report\n"
        f"Total records: {total}\n"
        f"Predicted attacks: {attacks}\n"
        f"Predicted normal: {normal}\n"
        f"High severity: {high}\n"
        "Predictions are educational model outputs, not confirmed incidents.\n"
    )

    st.download_button(
        "Download summary TXT",
        summary,
        file_name="threat_summary.txt",
        mime="text/plain"
    )

else:
    st.info("Click Run Threat Detection to analyze the records.")
    st.write("Expected features:")
    st.code(", ".join(FEATURES))
    st.warning(
        "Synthetic data is for testing only. Accuracy on synthetic "
        "data does not establish real-world detection performance."
    )