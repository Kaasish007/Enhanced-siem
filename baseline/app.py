from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mini_siem.inference import run_inference
from mini_siem.model import INPUT_DIM, load_autoencoder

st.set_page_config(page_title="Mini SIEM", layout="wide")
st.title("🔐 Mini SIEM Log Analysis & Threat Detection")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT.parent / "models" / "autoencoder_beth.pth"


@st.cache_resource
def get_model():
    return load_autoencoder(MODEL_PATH)


model = get_model()

with st.sidebar:
    st.header("Detection Settings")
    threshold_pct = st.slider(
        "Threshold percentile (higher = fewer threats)",
        min_value=90.0,
        max_value=99.9,
        value=98.0,
        step=0.1,
        format="%.1f",
    )

    st.divider()
    st.subheader("Expected input")
    st.caption(f"The model uses {INPUT_DIM} numeric features. Column order matters.")
    with st.expander("How the app reads your CSV"):
        st.markdown(
            """
- Drops label-like columns if present: `sus`, `evil`, `attack`, `label`, `anomaly`
- Converts `args` and `stackAddresses` to string length (if present)
- Converts object/categorical columns to numeric codes
- Uses the first 14 columns after preprocessing (pads with zeros if fewer)
"""
        )


tab1, tab2 = st.tabs(["Detection Results", "Model Comparison & Stats"])

results_all = None
results_view = None
meta = None
raw_df = None

with tab1:
    uploaded_file = st.file_uploader("Upload Security Log CSV", type=["csv"])

    if not uploaded_file:
        st.info("Upload a CSV file to begin analysis.")
        st.caption("If you’re not sure about the expected format, download a template below.")
        template_cols = [f"f{i}" for i in range(INPUT_DIM)]
        template_df = pd.DataFrame([[0] * INPUT_DIM], columns=template_cols)
        st.download_button(
            label="Download CSV template",
            data=template_df.to_csv(index=False),
            file_name="mini_siem_template.csv",
            mime="text/csv",
        )
    elif model is None:
        st.error(f"Model not loaded. Expected: {MODEL_PATH}")
    else:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        with st.spinner("Running threat detection..."):
            try:
                results_all, meta = run_inference(
                    model=model, df=raw_df, threshold_pct=threshold_pct
                )
            except Exception as e:
                st.error(f"Detection failed. Error: {e}")
                st.stop()

        st.success(
            f"Done. Risk threshold at {meta.threshold_pct:.1f}th percentile "
            f"(value={meta.threshold:.4g})."
        )

        st.subheader("Preprocessing & Feature Columns")
        with st.expander("Show details"):
            st.write(
                {
                    "Rows": len(raw_df),
                    "Input columns (raw)": len(raw_df.columns),
                    "Used feature columns (after preprocessing)": len(meta.used_feature_columns),
                    "Used feature names": meta.used_feature_columns,
                }
            )

        choice = st.radio("Filter by Result:", ["All", "🔴 Threat", "🟢 Normal"], horizontal=True)
        results_view = results_all
        if choice != "All":
            results_view = results_view[results_view["Result"] == choice]

        total_threats = int((results_view["Status"] == "Threat").sum())
        total_normals = int((results_view["Status"] == "Normal").sum())
        total_events = len(results_view)

        st.subheader("Summary Counts")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", total_events)
        col2.metric("Threats Detected", total_threats)
        col3.metric("Normal Events", total_normals)

        st.subheader("Detection Results")
        st.dataframe(results_view, hide_index=True, use_container_width=True)

        st.download_button(
            label="Download Results",
            data=results_view.to_csv(index=False),
            file_name="threat_results.csv",
            mime="text/csv",
        )

with tab2:
    st.header("Model Comparison & Statistics")

    if results_all is not None and model is not None and meta is not None:
        total = len(results_all)
        threats = int(results_all["Anomaly"].sum())
        threat_rate = (threats / total) * 100 if total > 0 else 0.0

        st.subheader("Threat Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", total)
        col2.metric("Detected Threats", threats)
        col3.metric("Normal Events", total - threats)

        st.subheader("Threat Level")
        st.progress(threat_rate / 100)
        st.write(f"Threat Rate: **{threat_rate:.2f}%**")

        st.subheader("Threat vs Normal Distribution")
        st.bar_chart(results_all["Status"].value_counts())

        st.subheader("Severity Distribution")
        st.bar_chart(results_all["Severity"].value_counts())

        st.subheader("Anomaly Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(results_all["RiskScore"], bins=50)
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.subheader("Risk Score Pattern")
        fig2, ax2 = plt.subplots()
        ax2.scatter(range(len(results_all)), results_all["RiskScore"], alpha=0.3)
        ax2.set_xlabel("Event Index")
        ax2.set_ylabel("Risk Score")
        st.pyplot(fig2)

        st.subheader("Top Suspicious Events")
        top = results_all[results_all["Status"] == "Threat"].sort_values("RiskScore", ascending=False)
        st.dataframe(top.head(20), hide_index=True, use_container_width=True)
    else:
        st.info("Upload a CSV file in the first tab to see statistics here.")

    st.subheader("Model Comparison")
    comp_path = PROJECT_ROOT / "model_results.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        st.dataframe(comp, hide_index=True, use_container_width=True)
        st.markdown(
            """
### Model Explanation

**Isolation Forest**
- Tree-based anomaly detection
- Good baseline performance

**Local Outlier Factor**
- Density-based anomaly detection
- Struggled with highly imbalanced data

**Autoencoder (Selected Model)**
- Deep learning model
- Learns normal system behaviour
- Detects anomalies using reconstruction error

The Autoencoder performed best for detecting abnormal host behaviour patterns in SIEM logs.
"""
        )
    else:
        st.warning("model_results.csv not found")
