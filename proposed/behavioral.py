"""
Host behavioral baseline and deviation scoring module.
Calculates statistical baselines for host event frequency and process activity,
producing normalized deviation scores with saturation controls.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


class HostBehaviorBaseline:
    """
    Learns statistical behavioral baselines (mean, std) per host across event frequency signals.
    Calculates host behavioral deviation z-scores safely without producing Inf or NaN.
    """

    def __init__(self, epsilon: float = 1e-3, default_std_ratio: float = 0.2):
        self.epsilon = epsilon
        self.default_std_ratio = default_std_ratio
        self.host_stats: Dict[str, Dict[str, float]] = {}
        self.global_mu: float = 1.0
        self.global_sigma: float = 1.0
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, freq_col: str = "host_frequency_5m") -> HostBehaviorBaseline:
        """
        Fit host baseline mean and standard deviation on normal dataset / calibration batch.
        No labels are used.
        """
        if freq_col not in df.columns:
            self.is_fitted = True
            return self

        host_col = "hostName" if "hostName" in df.columns else None
        freq_vals = df[freq_col].values.astype(float)

        self.global_mu = float(np.mean(freq_vals)) if len(freq_vals) > 0 else 1.0
        self.global_sigma = float(np.std(freq_vals)) if len(freq_vals) > 0 else 1.0
        if self.global_sigma < self.epsilon:
            self.global_sigma = max(1.0, self.global_mu * self.default_std_ratio)

        if host_col is not None:
            for host, group in df.groupby(host_col):
                h_vals = group[freq_col].values.astype(float)
                mu = float(np.mean(h_vals))
                sigma = float(np.std(h_vals))
                if sigma < self.epsilon:
                    sigma = max(1.0, mu * self.default_std_ratio)

                self.host_stats[str(host)] = {"mu": mu, "sigma": sigma}

        self.is_fitted = True
        return self

    def calculate_deviation(
        self, df: pd.DataFrame, freq_col: str = "host_frequency_5m"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate raw z-score behavioral deviation and tanh-normalized deviation.
        
        Returns:
            behavioral_deviation_raw: z = |x - mu| / (sigma + eps)
            behavioral_deviation_norm: tanh(z / 3.0) in [0.0, 1.0)
        """
        n_rows = len(df)
        if n_rows == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        if freq_col not in df.columns:
            return np.zeros(n_rows, dtype=float), np.zeros(n_rows, dtype=float)

        if not self.is_fitted:
            # Fit on current df if not previously fitted
            self.fit(df, freq_col=freq_col)

        freq_vals = df[freq_col].values.astype(float)
        host_col = "hostName" if "hostName" in df.columns else None
        hosts = df[host_col].astype(str).values if host_col is not None else None

        raw_dev = np.zeros(n_rows, dtype=float)
        for i in range(n_rows):
            val = freq_vals[i]
            h_name = hosts[i] if hosts is not None else None
            stats = self.host_stats.get(h_name, {"mu": self.global_mu, "sigma": self.global_sigma}) if h_name else None

            if stats is None:
                mu, sigma = self.global_mu, self.global_sigma
            else:
                mu, sigma = stats["mu"], stats["sigma"]

            z = abs(val - mu) / (sigma + self.epsilon)
            raw_dev[i] = z

        # Robust tanh normalization to prevent extreme values from overwhelming the score
        norm_dev = np.tanh(raw_dev / 3.0)
        return raw_dev, norm_dev
