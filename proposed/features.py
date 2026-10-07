"""
Behavioral and temporal feature extraction module.
Extracts host-level event frequencies, process frequencies, temporal signals, and rarity metrics
without using ground-truth attack labels.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


LABEL_COLS = ["sus", "evil", "attack", "label", "anomaly"]


def extract_behavioral_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract host behavioral, temporal, and frequency features from an uploaded log DataFrame.
    
    IMPORTANT: Ground-truth labels (LABEL_COLS) are NOT used to construct features.
    
    Returns:
        enhanced_df: DataFrame with additional feature columns.
        metadata: Dict summarizing available features and column status.
    """
    out = df.copy()
    meta: Dict[str, Any] = {
        "has_host": "hostName" in df.columns,
        "has_process": "processName" in df.columns,
        "has_event": "eventName" in df.columns,
        "has_timestamp": "timestamp" in df.columns,
        "available_columns": list(df.columns),
    }

    n_rows = len(out)
    if n_rows == 0:
        out["host_frequency_1h"] = 0.0
        out["host_frequency_5m"] = 0.0
        out["process_frequency"] = 0.0
        out["event_frequency"] = 0.0
        out["process_rarity_score"] = 0.0
        out["relative_time_s"] = 0.0
        return out, meta

    # Handle Host feature
    if "hostName" in out.columns:
        hosts = out["hostName"].astype(str).fillna("unknown_host")
    else:
        hosts = pd.Series(["default_host"] * n_rows, index=out.index)
    out["_host"] = hosts

    # Handle Timestamp feature securely
    if "timestamp" in out.columns:
        ts_numeric = pd.to_numeric(out["timestamp"], errors="coerce")
        if ts_numeric.notna().sum() > 0:
            ts_filled = ts_numeric.fillna(ts_numeric.median() if ts_numeric.notna().sum() > 0 else 0.0)
            t_min = ts_filled.min()
            out["relative_time_s"] = ts_filled - t_min
        else:
            out["relative_time_s"] = np.arange(n_rows, dtype=float)
    else:
        out["relative_time_s"] = np.arange(n_rows, dtype=float)

    # Calculate Host Frequencies using actual timestamps & host grouping
    rel_time = out["relative_time_s"].values
    host_freq_5m = np.zeros(n_rows, dtype=float)
    host_freq_1h = np.zeros(n_rows, dtype=float)

    # Calculate rolling window event frequencies per host
    for host_val, group_indices in out.groupby("_host").groups.items():
        idx_arr = np.array(list(group_indices))
        times = rel_time[idx_arr]

        # Use searchsorted to compute counts in window [t - window, t]
        # 5m = 300s, 1h = 3600s
        for k, i_loc in enumerate(idx_arr):
            t_curr = times[k]
            # events in [t_curr - 300, t_curr]
            idx_start_5m = np.searchsorted(times, t_curr - 300.0, side="left")
            count_5m = k - idx_start_5m + 1
            host_freq_5m[i_loc] = float(count_5m)

            # events in [t_curr - 3600, t_curr]
            idx_start_1h = np.searchsorted(times, t_curr - 3600.0, side="left")
            count_1h = k - idx_start_1h + 1
            host_freq_1h[i_loc] = float(count_1h)

    out["host_frequency_5m"] = host_freq_5m
    out["host_frequency_1h"] = host_freq_1h

    # Calculate Process Frequency & Rarity per host when processName is available
    if "processName" in out.columns:
        proc_series = out["processName"].astype(str).fillna("unknown_proc")
        proc_counts = proc_series.value_counts(normalize=True)
        out["process_frequency"] = proc_series.map(proc_counts).fillna(0.0).astype(float)
        out["process_rarity_score"] = 1.0 - out["process_frequency"]
    else:
        out["process_frequency"] = 1.0
        out["process_rarity_score"] = 0.0

    # Calculate Event Frequency per host when eventName is available
    if "eventName" in out.columns:
        evt_series = out["eventName"].astype(str).fillna("unknown_evt")
        evt_counts = evt_series.value_counts(normalize=True)
        out["event_frequency"] = evt_series.map(evt_counts).fillna(0.0).astype(float)
    else:
        out["event_frequency"] = 1.0

    out.drop(columns=["_host"], errors="ignore", inplace=True)
    return out, meta
