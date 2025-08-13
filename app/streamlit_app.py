
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

st.set_page_config(page_title="Aero-PDM Dashboard", layout="wide")

st.title("Aero-PDM: Predictive Maintenance for Turbofan Engines")

rul_model_path = MODELS / "rul_random_forest.joblib"
anomaly_model_path = MODELS / "anomaly_isolation_forest.joblib"

if not (rul_model_path.exists() and anomaly_model_path.exists()):
    st.warning("Models not found. Please run training steps first.")
else:
    rul_model = joblib.load(rul_model_path)
    anom_model = joblib.load(anomaly_model_path)

    st.sidebar.header("Upload Feature CSV (optional)")
    uploaded = st.sidebar.file_uploader("Upload CSV with engineered features", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        default_path = DATA / "test_features.csv"
        if default_path.exists():
            df = pd.read_csv(default_path)
        else:
            st.stop()

    id_cols = ["unit","cycle"]
    target_cols = [c for c in ["RUL","failed"] if c in df.columns]
    drop_cols = id_cols + target_cols
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    pred_rul = rul_model.predict(X)
    anom_score = -anom_model.score_samples(X)

    viz = df[id_cols].copy()
    viz["pred_RUL"] = pred_rul
    if "RUL" in df.columns:
        viz["true_RUL"] = df["RUL"]
    viz["anomaly_score"] = anom_score

    st.subheader("Prediction Overview")
    st.dataframe(viz.head(100))

    st.subheader("RUL by Cycle (select unit)")
    units = sorted(viz["unit"].unique())
    unit_sel = st.selectbox("Engine Unit", units)
    plot_df = viz[viz["unit"] == unit_sel].sort_values("cycle")

    fig1 = px.line(plot_df, x="cycle", y=["pred_RUL"] + (["true_RUL"] if "true_RUL" in plot_df.columns else []),
                   title=f"Unit {unit_sel}: RUL over cycles")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Anomaly Score by Cycle")
    fig2 = px.line(plot_df, x="cycle", y="anomaly_score", title=f"Unit {unit_sel}: Anomaly score")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Higher anomaly score indicates more unusual behavior vs healthy baseline.")
