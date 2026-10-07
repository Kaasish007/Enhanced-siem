"""
Host-adaptive thresholding engine.
Calculates dynamic thresholds for reconstruction error that adjust based on host context and behavioral signals,
while strictly remaining scale-matched to the baseline ML reconstruction MSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdaptiveThresholdEngine:
    """
    Computes host-adaptive detection thresholds for PyTorch Autoencoder reconstruction MSE.
    
    The adaptive threshold T_adaptive is bounded around the baseline global threshold T_global,
    ensuring it never collapses to near-zero (preventing catastrophic false-positive saturation).
    """

    def __init__(self, sensitivity: float = 0.4, min_factor: float = 0.5, max_factor: float = 1.5):
        self.sensitivity = sensitivity
        self.min_factor = min_factor
        self.max_factor = max_factor

    def compute_adaptive_thresholds(
        self,
        risk_scores: np.ndarray,
        global_threshold: float,
        behavioral_dev_norm: np.ndarray,
        hosts: pd.Series = None,
    ) -> np.ndarray:
        """
        Compute per-event scale-matched adaptive thresholds.
        
        Args:
            risk_scores: Original Autoencoder reconstruction MSE array.
            global_threshold: Baseline percentile threshold calculated on risk_scores.
            behavioral_dev_norm: Normalized behavioral deviation in [0.0, 1.0).
            hosts: Optional hostName series for host-level threshold adjustment.
            
        Returns:
            adaptive_thresholds: Array of thresholds bounded in [min_factor * T_global, max_factor * T_global].
        """
        n = len(risk_scores)
        if n == 0:
            return np.array([], dtype=float)

        # Baseline check to ensure global_threshold is positive and finite
        if not np.isfinite(global_threshold) or global_threshold <= 0:
            global_threshold = float(np.percentile(risk_scores, 98.0)) if n > 0 else 1.0
            if global_threshold <= 0:
                global_threshold = 1.0

        # Adjust threshold based on normalized behavioral deviation z_norm
        # When z_norm > 0.5 (elevated activity), threshold tightens (smaller multiplier down to min_factor)
        # When z_norm < 0.5 (calm activity), threshold relaxes (larger multiplier up to max_factor)
        dev_shift = behavioral_dev_norm - 0.5
        factors = 1.0 - (self.sensitivity * dev_shift)
        factors = np.clip(factors, self.min_factor, self.max_factor)

        adaptive_thresholds = global_threshold * factors

        # Explicit finite validation check (Rule 21)
        assert np.all(np.isfinite(adaptive_thresholds)), "Adaptive thresholds must be finite!"
        assert np.all(adaptive_thresholds > 0), "Adaptive thresholds must be strictly positive!"

        return adaptive_thresholds.astype(float)
