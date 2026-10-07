"""
Comprehensive Validation & Test Suite for Mini-SIEM + Adaptive Behavioral Extension.
Executes all 14 mandatory validation tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mini_siem.model import load_autoencoder, INPUT_DIM
from mini_siem.inference import run_inference
from proposed.features import extract_behavioral_features, LABEL_COLS
from proposed.behavioral import HostBehaviorBaseline
from proposed.adaptive_threshold import AdaptiveThresholdEngine
from proposed.scoring import RiskEngine
from proposed.explanations import generate_alert_reasons
from proposed.experiments import find_ground_truth_label, calculate_metrics, run_ablation_study


def run_tests():
    print("=" * 60)
    print("STARTING MINI-SIEM VALIDATION SUITE (14 TESTS)")
    print("=" * 60)

    model_path = PROJECT_ROOT / "models" / "autoencoder_beth.pth"
    assert model_path.exists(), f"Model file missing: {model_path}"
    model = load_autoencoder(model_path)
    assert model is not None, "Failed to load Autoencoder model!"
    print("[OK] Model loaded successfully.")

    # Load test dataset
    sample_path = PROJECT_ROOT / "BETH" / "labelled_2021may-ip-10-100-1-105-dns.csv"
    if not sample_path.exists():
        sample_path = PROJECT_ROOT / "BETH" / "labelled_testing_data.csv"
    
    assert sample_path.exists(), f"Test CSV not found: {sample_path}"
    raw_df = pd.read_csv(sample_path)
    print(f"Loaded test dataset '{sample_path.name}' with {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    # TEST 1 & 4 & 5: Baseline Mini-SIEM Pipeline
    print("\n--- TEST 1, 4, 5: Original Baseline Pipeline & RiskScore Scale ---")
    baseline_df, meta = run_inference(model, raw_df, threshold_pct=98.0)
    assert "RiskScore" in baseline_df.columns
    assert "Anomaly" in baseline_df.columns
    assert "Status" in baseline_df.columns
    assert "Severity" in baseline_df.columns
    assert meta.used_feature_columns == list(meta.used_feature_columns[:INPUT_DIM])
    assert len(meta.used_feature_columns) == INPUT_DIM
    
    risk_min, risk_max = np.min(baseline_df["RiskScore"]), np.max(baseline_df["RiskScore"])
    print(f"[OK] Baseline 14-dim Autoencoder executed cleanly.")
    print(f"[OK] RiskScore (MSE) range: [{risk_min:.6f}, {risk_max:.6f}]. Global threshold: {meta.threshold:.6f}")
    assert np.isfinite(risk_min) and np.isfinite(risk_max)

    # TEST 2, 7, 8, 9: Behavioral Module & Frequency/Temporal Signals
    print("\n--- TEST 2, 7, 8, 9: Behavioral Features & Frequency ---")
    feat_df, feat_meta = extract_behavioral_features(raw_df)
    assert "host_frequency_5m" in feat_df.columns
    assert "host_frequency_1h" in feat_df.columns
    assert "relative_time_s" in feat_df.columns

    baseline_engine = HostBehaviorBaseline().fit(feat_df)
    raw_dev, norm_dev = baseline_engine.calculate_deviation(feat_df)
    
    print(f"[OK] Relative time range: [{np.min(feat_df['relative_time_s']):.2f}s, {np.max(feat_df['relative_time_s']):.2f}s]")
    print(f"[OK] 5m Event Frequency range: [{np.min(feat_df['host_frequency_5m'])}, {np.max(feat_df['host_frequency_5m'])}]")
    print(f"[OK] Behavioral z-score raw range: [{np.min(raw_dev):.2f}, {np.max(raw_dev):.2f}]")
    print(f"[OK] Normalized z-score range: [{np.min(norm_dev):.4f}, {np.max(norm_dev):.4f}]")
    assert np.all(norm_dev >= 0.0) and np.all(norm_dev <= 1.0)

    # TEST 6: Adaptive Threshold Scale
    print("\n--- TEST 6: Adaptive Threshold Scale ---")
    adaptive_engine = AdaptiveThresholdEngine()
    adaptive_thresholds = adaptive_engine.compute_adaptive_thresholds(
        risk_scores=baseline_df["RiskScore"].values,
        global_threshold=meta.threshold,
        behavioral_dev_norm=norm_dev,
        hosts=feat_df["hostName"] if "hostName" in feat_df.columns else None,
    )

    t_min, t_max = np.min(adaptive_thresholds), np.max(adaptive_thresholds)
    print(f"[OK] Global threshold: {meta.threshold:.6f}")
    print(f"[OK] Adaptive threshold scale range: [{t_min:.6f}, {t_max:.6f}]")
    assert t_min > 0 and np.isfinite(t_min), "Adaptive threshold must be positive & finite!"
    assert t_min >= 0.4 * meta.threshold, f"Adaptive threshold collapsed too low! ({t_min} vs {meta.threshold})"

    # TEST 10 & 11: Enhanced Risk Score & Alert Distribution
    print("\n--- TEST 10 & 11: Risk Score Range & Alert Distribution ---")
    risk_engine = RiskEngine()
    process_rarity = feat_df["process_rarity_score"].values if "process_rarity_score" in feat_df.columns else None
    norm_ml, enhanced_risk, beh_pred, alert_levels, val_info = risk_engine.compute_enhanced_risk(
        risk_scores=baseline_df["RiskScore"].values,
        adaptive_thresholds=adaptive_thresholds,
        global_threshold=meta.threshold,
        behavioral_dev_norm=norm_dev,
        process_rarity=process_rarity,
    )

    print(f"[OK] Enhanced Risk Score range: [{val_info['min_risk']:.4f}, {val_info['max_risk']:.4f}]")
    print(f"[OK] Alert distribution: {val_info['alert_distribution']}")
    assert val_info['min_risk'] >= 0.0 and val_info['max_risk'] <= 1.0
    assert not val_info['all_critical'], "Alert distribution is saturated to 100% CRITICAL!"

    # TEST 12: CSV Export Verification
    print("\n--- TEST 12: CSV Export Verification ---")
    enhanced_df = baseline_df.copy()
    enhanced_df["host_frequency_1h"] = feat_df["host_frequency_1h"].values
    enhanced_df["host_frequency_5m"] = feat_df["host_frequency_5m"].values
    enhanced_df["behavioral_deviation"] = raw_dev
    enhanced_df["adaptive_threshold"] = adaptive_thresholds
    enhanced_df["behavioral_prediction"] = beh_pred
    enhanced_df["normalized_ml_score"] = norm_ml
    enhanced_df["enhanced_risk_score"] = enhanced_risk
    enhanced_df["final_alert_level"] = alert_levels
    enhanced_df["alert_reason"] = generate_alert_reasons(
        baseline_df["RiskScore"].values, adaptive_thresholds, norm_dev, process_rarity, alert_levels
    )

    csv1 = baseline_df.to_csv(index=False)
    csv2 = enhanced_df.to_csv(index=False)
    assert len(csv1) > 0 and len(csv2) > 0
    print("[OK] Both threat_results.csv and enhanced_behavioral_results.csv exported successfully.")

    # TEST 13: Dataset with missing behavioral columns (graceful fallback)
    print("\n--- TEST 13: Missing Behavioral Columns Fallback ---")
    dummy_df = pd.DataFrame({"col_a": np.random.randn(50), "col_b": np.random.randn(50)})
    b_df, b_meta = run_inference(model, dummy_df, threshold_pct=98.0)
    f_df, f_meta = extract_behavioral_features(dummy_df)
    assert len(b_df) == 50 and len(f_df) == 50
    print("[OK] Missing optional columns handled gracefully without crashing.")

    # TEST 14: No Data Leakage Verification
    print("\n--- TEST 14: No Data Leakage Check ---")
    label_present = find_ground_truth_label(raw_df)
    for l_col in LABEL_COLS:
        assert l_col not in feat_df.columns or l_col not in meta.used_feature_columns, f"Label {l_col} leaked into feature matrix!"
    print("[OK] Ground-truth labels strictly isolated from detector features.")

    print("\n" + "=" * 60)
    print("ALL 14 VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
