"""
Explainability & Alert Reason generator module.
Provides clear, human-readable descriptions of contributing threat signals for security analysts.
"""

from __future__ import annotations

from typing import List
import numpy as np


def generate_alert_reasons(
    risk_scores: np.ndarray,
    adaptive_thresholds: np.ndarray,
    behavioral_dev_norm: np.ndarray,
    process_rarity: np.ndarray = None,
    alert_levels: List[str] = None,
) -> List[str]:
    """
    Generate natural language alert reasons based on contributing detection signals.
    """
    n = len(risk_scores)
    if n == 0:
        return []

    if process_rarity is None:
        process_rarity = np.zeros(n, dtype=float)

    reasons: List[str] = []
    for i in range(n):
        ml_exceeded = risk_scores[i] > adaptive_thresholds[i]
        b_dev = behavioral_dev_norm[i]
        rarity = process_rarity[i]
        level = alert_levels[i] if alert_levels and i < len(alert_levels) else "LOW"

        signals = []
        if ml_exceeded:
            signals.append("Autoencoder anomaly score exceeded host-adaptive threshold")
        if b_dev >= 0.50:
            signals.append("Event frequency is significantly above historical host baseline")
        if rarity >= 0.80:
            signals.append("Rare process execution detected on this host")

        if len(signals) == 0:
            if level in ["HIGH", "CRITICAL"]:
                reasons.append("Elevated composite risk score across host behavioral metrics.")
            else:
                reasons.append("Normal system activity conforming to baseline patterns.")
        elif len(signals) == 1:
            reasons.append(signals[0] + ".")
        elif len(signals) == 2:
            reasons.append(f"{signals[0]} and {signals[1].lower()}.")
        else:
            reasons.append("Multiple contributing signals: High ML anomaly score, elevated host activity rate, and rare process execution.")

    return reasons
