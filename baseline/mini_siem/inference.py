from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .model import INPUT_DIM


LABEL_COLS = ["sus", "evil", "attack", "label", "anomaly"]


def preprocess_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an uploaded log-event CSV into a purely-numeric feature matrix that the model can consume.
    """
    df = df.copy()
    df = df.drop(columns=[c for c in LABEL_COLS if c in df.columns], errors="ignore")

    # Turn potentially high-cardinality string fields into something numeric.
    if "args" in df.columns:
        df["args"] = df["args"].astype(str).str.len()
    if "stackAddresses" in df.columns:
        df["stackAddresses"] = df["stackAddresses"].astype(str).str.len()

    # Encode categorical/text columns to integer codes.
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    # Final numeric coercion; invalid values become 0.
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


@dataclass(frozen=True)
class InferenceMeta:
    threshold: float
    threshold_pct: float
    used_feature_columns: List[str]
    p99: float
    p97: float


def _align_features(features: pd.DataFrame, input_dim: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure exactly `input_dim` columns by taking the first N columns (or padding with zeros).
    Column order matters.
    """
    if features.shape[1] > input_dim:
        used = list(features.columns[:input_dim])
        return features.iloc[:, :input_dim].copy(), used

    if features.shape[1] < input_dim:
        missing = input_dim - features.shape[1]
        for i in range(missing):
            features[f"pad_{i}"] = 0
        used = list(features.columns[:input_dim])
        return features.iloc[:, :input_dim].copy(), used

    used = list(features.columns[:input_dim])
    return features.copy(), used


def run_inference(
    model: nn.Module,
    df: pd.DataFrame,
    threshold_pct: float = 98.0,
    input_dim: int = INPUT_DIM,
) -> Tuple[pd.DataFrame, InferenceMeta]:
    """
    Run autoencoder reconstruction error and attach risk/anomaly columns.
    """
    features = preprocess_logs(df)
    features, used_feature_columns = _align_features(features, input_dim=input_dim)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values.astype(np.float32))

    X_tensor = torch.tensor(X_scaled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        recon = model(X_tensor)
        error = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()

    threshold = float(np.percentile(error, threshold_pct))
    p99 = float(np.percentile(error, 99))
    p97 = float(np.percentile(error, 97))

    status = np.where(error > threshold, "Threat", "Normal")
    result = np.where(status == "Threat", "🔴 Threat", "🟢 Normal")

    def severity(score: float) -> str:
        if score > p99:
            return "High"
        if score > p97:
            return "Medium"
        return "Low"

    out = df.copy()
    out["RiskScore"] = error
    out["Anomaly"] = error > threshold
    out["Status"] = status
    out["Result"] = result
    out["Severity"] = [severity(s) for s in error]

    meta = InferenceMeta(
        threshold=threshold,
        threshold_pct=float(threshold_pct),
        used_feature_columns=used_feature_columns,
        p99=p99,
        p97=p97,
    )
    return out, meta

