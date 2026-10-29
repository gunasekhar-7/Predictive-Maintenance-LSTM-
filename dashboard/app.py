"""Streamlit dashboard for RUL health monitoring and alerting.

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

st.set_page_config(page_title="Predictive Maintenance - RUL Monitor", layout="wide")

st.title("Predictive Maintenance — Remaining Useful Life Monitor")
st.caption("LSTM-based RUL prediction on the NASA C-MAPSS turbofan dataset.")

MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
ALERT_THRESHOLD = 30.0


@st.cache_resource
def load_artifacts():
    meta_path = MODEL_DIR / "model_meta.json"
    scaler_path = MODEL_DIR / "scaler.pkl"
    columns_path = MODEL_DIR / "feature_columns.json"
    scripted_path = MODEL_DIR / "best_model_scripted.pt"

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    feature_columns = json.loads(columns_path.read_text()) if columns_path.exists() else []

    model = None
    if scripted_path.exists():
        model = torch.jit.load(str(scripted_path))
        model.eval()

    return meta, scaler, feature_columns, model


meta, scaler, feature_columns, model = load_artifacts()

window_length = int(meta.get("window_length", 30))
architecture = meta.get("architecture", "unknown")

st.sidebar.header("Model info")
st.sidebar.write(f"Architecture: {architecture}")
st.sidebar.write(f"Window length: {window_length}")
st.sidebar.write(f"Features: {len(feature_columns)}")
if meta.get("best_val_rmse") is not None:
    st.sidebar.write(f"Best val RMSE: {meta['best_val_rmse']:.2f}")

alert_threshold = st.sidebar.number_input(
    "Alert threshold (cycles)", min_value=1.0, max_value=200.0, value=ALERT_THRESHOLD, step=5.0
)

st.sidebar.divider()
st.sidebar.subheader("Input sensor readings")
uploaded = st.sidebar.file_uploader("Upload engine readings CSV", type=["csv"])

if uploaded is not None:
    import pandas as pd

    df = pd.read_csv(uploaded)
    st.subheader("Uploaded data preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Try to align columns to feature_columns
    if feature_columns and all(c in df.columns for c in feature_columns):
        readings = df[feature_columns].values.astype(float)
    else:
        # Use first N numeric columns
        numeric = df.select_dtypes(include=[np.number]).values.astype(float)
        readings = numeric

    if model is not None and scaler is not None:
        n = readings.shape[0]
        if n < window_length:
            pad = np.zeros((window_length - n, readings.shape[1]), dtype=float)
            readings = np.vstack([pad, readings])
        else:
            readings = readings[-window_length:]

        readings_scaled = scaler.transform(readings)
        x = torch.from_numpy(readings_scaled.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).cpu().numpy().squeeze()
        predicted_rul = float(pred)

        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted RUL (cycles)", f"{predicted_rul:.1f}")
        col2.metric("Window length", window_length)
        alert = predicted_rul < alert_threshold
        col3.metric("Status", "ALERT" if alert else "OK", delta=None)

        if alert:
            st.error(f"MAINTENANCE ALERT: Predicted RUL {predicted_rul:.1f} < threshold {alert_threshold}")
        else:
            st.success(f"Engine healthy. Predicted RUL {predicted_rul:.1f} >= threshold {alert_threshold}.")

        # Health curve (simulated: show trailing actual if a 'rul' column exists, else predicted trend)
        if "rul_capped" in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(df))),
                    y=df["rul_capped"],
                    mode="lines",
                    name="Actual RUL",
                    line=dict(color="#1f77b4", width=2),  # noqa: C408
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[len(df) - 1],
                    y=[predicted_rul],
                    mode="markers",
                    name="Predicted RUL",
                    marker=dict(color="#ff7f0e", size=12, symbol="diamond"),  # noqa: C408
                )
            )
            fig.add_hline(y=alert_threshold, line_dash="dot", line_color="#d62728", annotation_text="Alert threshold")
            fig.update_layout(title="Engine Health Curve", xaxis_title="Cycle", yaxis_title="RUL", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model or scaler not found. Train the model first (see README).")
else:
    st.info("Upload a CSV of sensor readings to get a RUL prediction and health assessment.")
    st.markdown(
        """
        **Expected CSV columns:** one row per engine cycle, with the normalized sensor columns used in training
        (see `models/feature_columns.json`). Optionally include a `rul_capped` column to plot the actual health curve.

        **API:** POST `http://localhost:8000/predict` with a JSON body:
        ```json
        {
          "readings": [[...feature values...], [...], ...],
          "feature_columns": [...]
        }
        ```
        """
    )
