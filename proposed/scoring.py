"""
Adaptive Risk Scoring Engine & Alert Classifier.
Combines normalized Autoencoder reconstruction errors with behavioral deviation and process rarity signals,
enforcing strict numerical bounds and plausible alert distribution checks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class RiskEngine:
    """
    Combines baseline ML reconstruction error and parallel behavioral signals into an Enhanced Risk Score.
    """

    def __init__(
        self,
        w_ml: float = 0.40,
        w_beh: float = 0.30,
        w_rare: float = 0.15,
        w_exceed: float = 0.15,
    ):
        self.w_ml = w_ml
        self.w_beh = w_beh
        self.w_rare = w_rare
        self.w_exceed = w_exceed
        total_w = w_ml + w_beh + w_rare + w_exceed
        assert abs(total_w - 1.0) < 1e-4, f"Scoring weights must sum to 1.0, got {total_w}"

    def compute_enhanced_risk(
        self,
        risk_scores: np.ndarray,
        adaptive_thresholds: np.ndarray,
        global_threshold: float,
        behavioral_dev_norm: np.ndarray,
        process_rarity: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Calculates enhanced risk components and assigns final alert levels.
        
        Returns:
            normalized_ml_score: Sigmoid-mapped representation of original MSE in [0.0, 1.0].
            enhanced_risk_score: Combined risk score in [0.0, 1.0].
            behavioral_prediction: Boolean array indicating whether adaptive threshold was exceeded.
            final_alert_level: List of strings ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
            validation_info: Dict containing distribution statistics and health checks.
        """
        n = len(risk_scores)
        if n == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=bool),
                [],
                {},
            )

        if process_rarity is None:
            process_rarity = np.zeros(n, dtype=float)

        # 1. Validation checks (Rule 21)
        assert np.all(np.isfinite(risk_scores)), "RiskScore contains non-finite values (NaN/Inf)!"
        assert np.all(np.isfinite(adaptive_thresholds)), "adaptive_threshold contains non-finite values!"

        # 2. Normalized ML score: Sigmoid transform centered at global threshold
        std_score = float(np.std(risk_scores))
        if std_score < 1e-6:
            std_score = max(1.0, float(global_threshold) * 0.5)

        # Center at global threshold and scale smoothly
        z_ml = (risk_scores - global_threshold) / (std_score + 1e-4)
        normalized_ml_score = 1.0 / (1.0 + np.exp(-z_ml))
        normalized_ml_score = np.clip(normalized_ml_score, 0.0, 1.0)

        # 3. Exceedance flag: Reconstruction MSE > adaptive threshold
        exceed_flag = (risk_scores > adaptive_thresholds).astype(float)
        behavioral_prediction = risk_scores > adaptive_thresholds

        # 4. Enhanced Risk Score calculation (weighted sum)
        enhanced_risk_score = (
            (self.w_ml * normalized_ml_score)
            + (self.w_beh * behavioral_dev_norm)
            + (self.w_rare * process_rarity)
            + (self.w_exceed * exceed_flag)
        )
        enhanced_risk_score = np.clip(enhanced_risk_score, 0.0, 1.0)

        # Explicit bound assertion (Rule 21)
        assert np.all(enhanced_risk_score >= 0.0) and np.all(enhanced_risk_score <= 1.0), (
            "enhanced_risk_score must be strictly in [0.0, 1.0]!"
        )

        # 5. Alert Level Classification
        alert_levels: List[str] = []
        for i in range(n):
            score = enhanced_risk_score[i]
            exceeded = behavioral_prediction[i]
            b_dev = behavioral_dev_norm[i]

            if score >= 0.80 or (exceeded and b_dev >= 0.70):
                alert_levels.append("CRITICAL")
            elif score >= 0.60 or exceeded:
                alert_levels.append("HIGH")
            elif score >= 0.40:
                alert_levels.append("MEDIUM")
            else:
                alert_levels.append("LOW")

        # Validation stats check
        counts = pd.Series(alert_levels).value_counts().to_dict()
        val_info = {
            "min_risk": float(np.min(enhanced_risk_score)),
            "max_risk": float(np.max(enhanced_risk_score)),
            "mean_risk": float(np.mean(enhanced_risk_score)),
            "alert_distribution": counts,
            "all_critical": len(counts) == 1 and "CRITICAL" in counts,
            "all_anomalous": np.all(behavioral_prediction),
        }

        return (
            normalized_ml_score.astype(float),
            enhanced_risk_score.astype(float),
            behavioral_prediction,
            alert_levels,
            val_info,
        )
