from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mini_siem.inference import run_inference
from mini_siem.model import INPUT_DIM, load_autoencoder
from proposed.adaptive_threshold import AdaptiveThresholdEngine
from proposed.behavioral import HostBehaviorBaseline
from proposed.experiments import find_ground_truth_label, calculate_metrics, run_ablation_study
from proposed.explanations import generate_alert_reasons
from proposed.features import extract_behavioral_features
from proposed.scoring import RiskEngine

st.set_page_config(page_title="Mini SIEM + Adaptive Behavioral Extension", layout="wide")
st.title("🔐 Mini SIEM Log Analysis & Adaptive Behavioral Threat Detection")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder_beth.pth"


@st.cache_resource
def get_model():
    return load_autoencoder(MODEL_PATH)


model = get_model()

# Sidebar Setup
with st.sidebar:
    st.header("Detection Mode & Controls")
    detection_mode = st.radio(
        "Select Detection Mode:",
        [
            "Original Mini-SIEM (Baseline)",
            "Enhanced Behavioral SIEM (Proposed)",
            "Comparative Research View",
        ],
        index=1,
    )

    st.divider()
    st.subheader("Threshold Settings")
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
    st.caption(f"The model uses {INPUT_DIM} numeric features for the baseline Autoencoder. Column order matters.")
    with st.expander("How the app reads your CSV"):
        st.markdown(
            """
- Drops label-like columns if present: `sus`, `evil`, `attack`, `label`, `anomaly` for ML feature matrix.
- Converts `args` and `stackAddresses` to string length (if present).
- Converts object/categorical columns to numeric codes.
- Uses the first 14 columns after preprocessing (pads with zeros if fewer) for baseline Autoencoder.
- **Parallel Behavioral Pipeline**: Extracts `hostName`, `timestamp`, `processName`, and `eventName` for host behavioral profiling, event frequency, adaptive thresholding, and risk scoring.
"""
        )

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Detection Results",
    "Model Comparison & Stats",
    "Behavioral & Host Analysis",
    "Research Experiments",
])

baseline_df = None
enhanced_df = None
meta = None
raw_df = None
val_info = None

# Main Upload & Pipeline Execution
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

        with st.spinner("Running threat detection & behavioral analysis..."):
            try:
                # 1. Baseline Mini-SIEM Pipeline (100% UNTOUCHED)
                baseline_df, meta = run_inference(
                    model=model, df=raw_df, threshold_pct=threshold_pct
                )

                # 2. Parallel Behavioral Module Execution
                feat_df, feat_meta = extract_behavioral_features(raw_df)
                baseline_engine = HostBehaviorBaseline().fit(feat_df)
                raw_dev, norm_dev = baseline_engine.calculate_deviation(feat_df)

                # 3. Adaptive Threshold Calculation
                adaptive_engine = AdaptiveThresholdEngine()
                adaptive_thresholds = adaptive_engine.compute_adaptive_thresholds(
                    risk_scores=baseline_df["RiskScore"].values,
                    global_threshold=meta.threshold,
                    behavioral_dev_norm=norm_dev,
                    hosts=feat_df["hostName"] if "hostName" in feat_df.columns else None,
                )

                # 4. Enhanced Risk Engine
                risk_engine = RiskEngine()
                process_rarity = feat_df["process_rarity_score"].values if "process_rarity_score" in feat_df.columns else None
                norm_ml, enhanced_risk, beh_pred, alert_levels, val_info = risk_engine.compute_enhanced_risk(
                    risk_scores=baseline_df["RiskScore"].values,
                    adaptive_thresholds=adaptive_thresholds,
                    global_threshold=meta.threshold,
                    behavioral_dev_norm=norm_dev,
                    process_rarity=process_rarity,
                )

                # 5. Alert Reasons Generator
                alert_reasons = generate_alert_reasons(
                    risk_scores=baseline_df["RiskScore"].values,
                    adaptive_thresholds=adaptive_thresholds,
                    behavioral_dev_norm=norm_dev,
                    process_rarity=process_rarity,
                    alert_levels=alert_levels,
                )

                # Assemble Enhanced Dataframe (Preserving all original + baseline columns)
                enhanced_df = baseline_df.copy()
                enhanced_df["host_frequency_1h"] = feat_df["host_frequency_1h"].values
                enhanced_df["host_frequency_5m"] = feat_df["host_frequency_5m"].values
                enhanced_df["process_frequency"] = feat_df["process_frequency"].values
                enhanced_df["behavioral_deviation"] = raw_dev
                enhanced_df["behavioral_deviation_norm"] = norm_dev
                enhanced_df["adaptive_threshold"] = adaptive_thresholds
                enhanced_df["behavioral_prediction"] = beh_pred
                enhanced_df["normalized_ml_score"] = norm_ml
                enhanced_df["enhanced_risk_score"] = enhanced_risk
                enhanced_df["final_alert_level"] = alert_levels
                enhanced_df["alert_reason"] = alert_reasons

            except Exception as e:
                st.error(f"Detection failed. Error: {e}")
                st.stop()

        # Display Pipeline Mode Notice
        if detection_mode == "Original Mini-SIEM (Baseline)":
            st.success(
                f"Done [Original Baseline Mode]. Risk threshold at {meta.threshold_pct:.1f}th percentile "
                f"(value={meta.threshold:.4g})."
            )
        else:
            st.success(
                f"Done [Enhanced Behavioral SIEM Mode]. Global ML threshold at {meta.threshold_pct:.1f}th percentile "
                f"({meta.threshold:.4g}) with dynamic scale-matched host adaptive thresholding."
            )

        st.subheader("Preprocessing & Feature Columns")
        with st.expander("Show details"):
            st.write(
                {
                    "Rows": len(raw_df),
                    "Input columns (raw)": len(raw_df.columns),
                    "Baseline feature columns (14-dim Autoencoder)": len(meta.used_feature_columns),
                    "Used feature names": meta.used_feature_columns,
                    "Behavioral host columns detected": feat_meta,
                }
            )

        # Tab 1 Display Logic based on selected Mode
        if detection_mode == "Original Mini-SIEM (Baseline)":
            choice = st.radio("Filter by Result:", ["All", "🔴 Threat", "🟢 Normal"], horizontal=True)
            results_view = baseline_df.copy()
            if choice != "All":
                results_view = results_view[results_view["Result"] == choice]

            total_threats = int((results_view["Status"] == "Threat").sum())
            total_normals = int((results_view["Status"] == "Normal").sum())
            total_events = len(results_view)

            st.subheader("Summary Counts (Baseline Autoencoder)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Events", total_events)
            col2.metric("Threats Detected", total_threats)
            col3.metric("Normal Events", total_normals)

            st.subheader("Baseline Detection Results")
            st.dataframe(results_view, hide_index=True, use_container_width=True)

            st.download_button(
                label="Download Results (threat_results.csv)",
                data=results_view.to_csv(index=False),
                file_name="threat_results.csv",
                mime="text/csv",
            )
        else:
            choice = st.radio(
                "Filter by Enhanced Alert Level:",
                ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                horizontal=True,
            )
            results_view = enhanced_df.copy()
            if choice != "All":
                results_view = results_view[results_view["final_alert_level"] == choice]

            n_crit = int((enhanced_df["final_alert_level"] == "CRITICAL").sum())
            n_high = int((enhanced_df["final_alert_level"] == "HIGH").sum())
            n_med = int((enhanced_df["final_alert_level"] == "MEDIUM").sum())
            n_low = int((enhanced_df["final_alert_level"] == "LOW").sum())

            st.subheader("Summary Counts (Enhanced Adaptive Behavioral SIEM)")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Events", len(enhanced_df))
            col2.metric("Critical Alerts", n_crit)
            col3.metric("High Alerts", n_high)
            col4.metric("Medium Alerts", n_med)
            col5.metric("Low / Normal", n_low)

            st.subheader("Enhanced Detection Results")
            display_cols = list(raw_df.columns) + [
                "RiskScore", "Anomaly", "Status", "Severity",
                "adaptive_threshold", "behavioral_prediction", "enhanced_risk_score",
                "final_alert_level", "alert_reason",
            ]
            # Keep only existing columns
            display_cols = [c for c in display_cols if c in results_view.columns]
            st.dataframe(results_view[display_cols], hide_index=True, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.download_button(
                label="Download Baseline Results (threat_results.csv)",
                data=baseline_df.to_csv(index=False),
                file_name="threat_results.csv",
                mime="text/csv",
            )
            c2.download_button(
                label="Download Enhanced Behavioral Results (enhanced_behavioral_results.csv)",
                data=enhanced_df.to_csv(index=False),
                file_name="enhanced_behavioral_results.csv",
                mime="text/csv",
            )

# Tab 2: Model Comparison & Statistics
with tab2:
    st.header("Model Comparison & Statistics")

    if baseline_df is not None and model is not None and meta is not None:
        total = len(baseline_df)
        threats = int(baseline_df["Anomaly"].sum())
        threat_rate = (threats / total) * 100 if total > 0 else 0.0

        st.subheader("Threat Statistics (Baseline Autoencoder)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", total)
        col2.metric("Detected Threats", threats)
        col3.metric("Normal Events", total - threats)

        st.subheader("Threat Level")
        st.progress(min(1.0, max(0.0, threat_rate / 100)))
        st.write(f"Baseline Threat Rate: **{threat_rate:.2f}%**")

        st.subheader("Status & Severity Distribution")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Baseline Status Distribution")
            st.bar_chart(baseline_df["Status"].value_counts())
        with col_b:
            st.write("Baseline Severity Distribution")
            st.bar_chart(baseline_df["Severity"].value_counts())

        if enhanced_df is not None:
            st.subheader("Enhanced Alert Level Distribution")
            st.bar_chart(enhanced_df["final_alert_level"].value_counts())

        st.subheader("Anomaly Score Distribution (Baseline Reconstruction MSE)")
        fig, ax = plt.subplots()
        ax.hist(baseline_df["RiskScore"], bins=50, color="crimson", alpha=0.7, label="Baseline RiskScore (MSE)")
        ax.axvline(meta.threshold, color="black", linestyle="--", label=f"Global Threshold ({meta.threshold:.4f})")
        ax.set_xlabel("Reconstruction MSE (RiskScore)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Risk Score Pattern vs Event Index")
        fig2, ax2 = plt.subplots()
        ax2.scatter(range(len(baseline_df)), baseline_df["RiskScore"], alpha=0.4, c="crimson", label="RiskScore")
        if enhanced_df is not None:
            ax2.scatter(range(len(enhanced_df)), enhanced_df["adaptive_threshold"], alpha=0.3, c="blue", label="Adaptive Threshold")
        ax2.set_xlabel("Event Index")
        ax2.set_ylabel("Score Value")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Top Suspicious Events")
        top = baseline_df[baseline_df["Status"] == "Threat"].sort_values("RiskScore", ascending=False)
        st.dataframe(top.head(20), hide_index=True, use_container_width=True)
    else:
        st.info("Upload a CSV file in the first tab to see statistics here.")

    st.subheader("Original Model Comparison Benchmark")
    comp_path = PROJECT_ROOT / "model_results.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        st.dataframe(comp, hide_index=True, use_container_width=True)
        st.markdown(
            """
### Model Architecture & Benchmark Explanation

**Isolation Forest**
- Tree-based anomaly detection baseline.

**Local Outlier Factor**
- Density-based anomaly detection baseline.

**Autoencoder (Selected Baseline Model)**
- Deep learning reconstruction model (14 → 64 → 32 → 16 → 32 → 64 → 14).
- Learns normal system behavior; detects anomalies using reconstruction MSE (`RiskScore`).

**Enhanced Adaptive Behavioral SIEM (Proposed Layer)**
- Runs parallel host profiling, event frequency z-scores, and dynamic scale-matched adaptive thresholding on top of the Autoencoder.
"""
        )
    else:
        st.warning("model_results.csv not found")

# Tab 3: Behavioral & Host Analysis (NEW TAB)
with tab3:
    st.header("Behavioral & Host Analysis")

    if enhanced_df is not None:
        st.markdown("### Host Profiling & Frequency Signals")

        if "hostName" in enhanced_df.columns:
            host_counts = enhanced_df["hostName"].value_counts()
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.subheader("Active Host Distribution")
                st.dataframe(host_counts.reset_index().rename(columns={"index": "Host", "hostName": "Event Count"}), hide_index=True)
            with col_h2:
                st.subheader("Events Per Host Chart")
                st.bar_chart(host_counts)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("Host 5-Minute Window Event Rate")
            fig_f5, ax_f5 = plt.subplots()
            ax_f5.hist(enhanced_df["host_frequency_5m"], bins=30, color="teal", alpha=0.7)
            ax_f5.set_xlabel("Events in 5-min Window")
            ax_f5.set_ylabel("Count")
            st.pyplot(fig_f5)
        with col_b2:
            st.subheader("Behavioral Deviation z-Score Distribution")
            fig_z, ax_z = plt.subplots()
            ax_z.hist(enhanced_df["behavioral_deviation"], bins=30, color="purple", alpha=0.7)
            ax_z.set_xlabel("z-score Deviation")
            ax_z.set_ylabel("Frequency")
            st.pyplot(fig_z)

        st.subheader("Top Suspicious Behavioral Events")
        beh_top = enhanced_df.sort_values("enhanced_risk_score", ascending=False)
        display_beh_cols = [c for c in ["hostName", "processName", "eventName", "RiskScore", "adaptive_threshold", "behavioral_deviation", "enhanced_risk_score", "final_alert_level", "alert_reason"] if c in beh_top.columns]
        st.dataframe(beh_top[display_beh_cols].head(25), hide_index=True, use_container_width=True)

        st.subheader("Single Event Investigation")
        event_idx = st.number_input("Enter Event Index to Inspect:", min_value=0, max_value=len(enhanced_df)-1, value=0, step=1)
        target_row = enhanced_df.iloc[event_idx]

        st.info(f"**Alert Reason**: {target_row.get('alert_reason', 'N/A')}")
        c_i1, c_i2, c_i3, c_i4 = st.columns(4)
        c_i1.metric("RiskScore (MSE)", f"{target_row.get('RiskScore', 0.0):.4f}")
        c_i2.metric("Adaptive Threshold", f"{target_row.get('adaptive_threshold', 0.0):.4f}")
        c_i3.metric("Behavioral z-Score", f"{target_row.get('behavioral_deviation', 0.0):.2f} σ")
        c_i4.metric("Final Alert Level", f"{target_row.get('final_alert_level', 'LOW')}")

    else:
        st.info("Upload a CSV file in the first tab to inspect behavioral analysis.")

# Tab 4: Research Experiments (NEW TAB)
with tab4:
    st.header("Research Experiments & Comparative View")

    if baseline_df is not None and enhanced_df is not None:
        y_true = find_ground_truth_label(raw_df)

        st.subheader("1. Comparative Research View (Baseline vs Proposed)")

        base_metrics = calculate_metrics(y_true, baseline_df["Anomaly"].values, baseline_df["RiskScore"].values)
        prop_metrics = calculate_metrics(y_true, enhanced_df["behavioral_prediction"].values, enhanced_df["enhanced_risk_score"].values)

        if base_metrics["Evaluated"] and prop_metrics["Evaluated"]:
            st.success("Ground-truth attack labels found in dataset. Calculated empirical metrics below:")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("#### BASELINE (Original Mini-SIEM)")
                st.write(f"**Threshold**: Global {meta.threshold_pct:.1f}th Percentile ({meta.threshold:.4f})")
                st.write(f"**Detected Threats**: {int(baseline_df['Anomaly'].sum())}")
                st.write(f"**Precision**: {base_metrics['Precision']:.4f}")
                st.write(f"**Recall**: {base_metrics['Recall']:.4f}")
                st.write(f"**F1 Score**: {base_metrics['F1']:.4f}")
                st.write(f"**FPR**: {base_metrics['FPR']:.4f}")
                st.write(f"**FNR**: {base_metrics['FNR']:.4f}")

            with col_res2:
                st.markdown("#### PROPOSED (Adaptive Behavioral SIEM)")
                st.write(f"**Adaptive Threshold**: Dynamic scale-matched range ({np.min(enhanced_df['adaptive_threshold']):.4f} – {np.max(enhanced_df['adaptive_threshold']):.4f})")
                st.write(f"**Detected Threats**: {int(enhanced_df['behavioral_prediction'].sum())}")
                st.write(f"**Precision**: {prop_metrics['Precision']:.4f}")
                st.write(f"**Recall**: {prop_metrics['Recall']:.4f}")
                st.write(f"**F1 Score**: {prop_metrics['F1']:.4f}")
                st.write(f"**FPR**: {prop_metrics['FPR']:.4f}")
                st.write(f"**FNR**: {prop_metrics['FNR']:.4f}")

            st.markdown("#### Quantitative Comparison Table")
            comp_table = pd.DataFrame([
                {
                    "Metric": "Precision",
                    "Baseline": f"{base_metrics['Precision']:.4f}",
                    "Proposed": f"{prop_metrics['Precision']:.4f}",
                    "Difference": f"{prop_metrics['Precision'] - base_metrics['Precision']:+.4f}",
                },
                {
                    "Metric": "Recall",
                    "Baseline": f"{base_metrics['Recall']:.4f}",
                    "Proposed": f"{prop_metrics['Recall']:.4f}",
                    "Difference": f"{prop_metrics['Recall'] - base_metrics['Recall']:+.4f}",
                },
                {
                    "Metric": "F1-Score",
                    "Baseline": f"{base_metrics['F1']:.4f}",
                    "Proposed": f"{prop_metrics['F1']:.4f}",
                    "Difference": f"{prop_metrics['F1'] - base_metrics['F1']:+.4f}",
                },
                {
                    "Metric": "FPR (False Positive Rate)",
                    "Baseline": f"{base_metrics['FPR']:.4f}",
                    "Proposed": f"{prop_metrics['FPR']:.4f}",
                    "Difference": f"{prop_metrics['FPR'] - base_metrics['FPR']:+.4f}",
                },
                {
                    "Metric": "FNR (False Negative Rate)",
                    "Baseline": f"{base_metrics['FNR']:.4f}",
                    "Proposed": f"{prop_metrics['FNR']:.4f}",
                    "Difference": f"{prop_metrics['FNR'] - base_metrics['FNR']:+.4f}",
                },
            ])
            st.dataframe(comp_table, hide_index=True, use_container_width=True)
        else:
            st.warning("Ground-truth attack labels (sus, evil, attack, label, anomaly) were not detected in the uploaded CSV. Evaluation metrics are shown as N/A per research integrity rules.")
            st.table(pd.DataFrame([
                {"Metric": "Precision", "Baseline": "N/A", "Proposed": "N/A", "Difference": "N/A"},
                {"Metric": "Recall", "Baseline": "N/A", "Proposed": "N/A", "Difference": "N/A"},
                {"Metric": "F1-Score", "Baseline": "N/A", "Proposed": "N/A", "Difference": "N/A"},
                {"Metric": "FPR", "Baseline": "N/A", "Proposed": "N/A", "Difference": "N/A"},
                {"Metric": "FNR", "Baseline": "N/A", "Proposed": "N/A", "Difference": "N/A"},
            ]))

        st.subheader("2. Systematic 6-Model Ablation Study")
        ablation_df = run_ablation_study(
            df=raw_df,
            baseline_anomalies=baseline_df["Anomaly"].values,
            baseline_risk_scores=baseline_df["RiskScore"].values,
            enhanced_prediction=enhanced_df["behavioral_prediction"].values,
            enhanced_risk_scores=enhanced_df["enhanced_risk_score"].values,
            behavioral_dev_norm=enhanced_df["behavioral_deviation_norm"].values,
            adaptive_thresholds=enhanced_df["adaptive_threshold"].values,
            global_threshold=meta.threshold,
        )
        st.dataframe(ablation_df, hide_index=True, use_container_width=True)
    else:
        st.info("Upload a CSV file in the first tab to view comparative research experiments.")
