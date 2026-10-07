import pandas as pd
import numpy as np
import os
import joblib

class HostBehaviorBaseline:
    def __init__(self, frequency_window_s=60):
        # We bin events by this window size to learn normal frequency
        self.frequency_window_s = frequency_window_s
        self.host_stats = {}
        self.is_fitted = False
        self.epsilon = 1e-4

    def fit(self, df: pd.DataFrame):
        """
        Expects a DataFrame with 'hostName' and 'timestamp'. 
        Extracts mu and sigma for event counts per frequency_window_s.
        """
        if "hostName" not in df.columns or "timestamp" not in df.columns:
            raise KeyError("DataFrame must contain 'hostName' and 'timestamp'")

        df = df.copy()
        # Sort chronologically
        df = df.sort_values(by="timestamp")

        # Group by host
        for host, group in df.groupby("hostName"):
            timestamps = group["timestamp"].values
            if len(timestamps) < 2:
                self.host_stats[host] = {"mu": 1.0, "sigma": 0.0}
                continue

            t_min = timestamps[0]
            t_max = timestamps[-1]
            
            # Avoid single bin issues
            if t_max - t_min < self.frequency_window_s:
                count = len(timestamps)
                self.host_stats[host] = {"mu": count, "sigma": count * 0.1}
                continue

            # Bin events into frequency_window_s
            bins = np.arange(t_min, t_max + self.frequency_window_s, self.frequency_window_s)
            counts, _ = np.histogram(timestamps, bins=bins)
            
            mu = np.mean(counts)
            sigma = np.std(counts)
            
            self.host_stats[host] = {
                "mu": float(mu),
                "sigma": float(sigma)
            }
            
        self.is_fitted = True

    def calculate_deviation(self, host: str, current_frequency: float) -> float:
        """
        Implementation of: Deviation = |f_current - mu_historical| / (sigma_historical + epsilon)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
            
        stats = self.host_stats.get(host, {"mu": 0.0, "sigma": 0.0})
        mu = stats["mu"]
        sigma = stats["sigma"]
        
        # If we've never seen the host or it had zero variance, penalize slightly but rationally
        if mu == 0.0 and sigma == 0.0:
            return float(current_frequency)
            
        dev = abs(current_frequency - mu) / (sigma + self.epsilon)
        return float(dev)
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "host_stats": self.host_stats,
            "frequency_window_s": self.frequency_window_s,
            "is_fitted": self.is_fitted,
            "epsilon": self.epsilon
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        state = joblib.load(filepath)
        obj = cls(frequency_window_s=state["frequency_window_s"])
        obj.host_stats = state["host_stats"]
        obj.is_fitted = state["is_fitted"]
        obj.epsilon = state["epsilon"]
        return obj
