import numpy as np
import os
import joblib

class AdaptiveThresholding:
    def __init__(self, k_sensitivity=0.5):
        self.k_sensitivity = k_sensitivity
        self.global_threshold = 0.0
        self.host_thresholds = {}
        self.is_fitted = False

    def fit(self, hosts: np.ndarray, errors: np.ndarray, base_pct=98.0):
        """
        Calculates T_global and T_host baselines from validation set errors.
        """
        if len(errors) == 0:
            return
            
        self.global_threshold = float(np.percentile(errors, base_pct))
        
        unique_hosts = set(hosts)
        for h in unique_hosts:
            host_errors = errors[hosts == h]
            if len(host_errors) > 10:
                self.host_thresholds[h] = float(np.percentile(host_errors, base_pct))
            else:
                self.host_thresholds[h] = self.global_threshold
                
        self.is_fitted = True

    def calculate_threshold(self, host: str, behavior_dev: float, temporal_dev: float) -> float:
        """
        T_adaptive = T_base / (1 + k * combined_dev)
        where combined_dev scales based on how unusual the activity is.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
            
        t_base = self.host_thresholds.get(host, self.global_threshold)
        
        # Combine deviations. If deviation > 1.0, it's anomalous.
        # Temporal dev is 0.0 to 1.0. Behavior dev can be 0.0 to Infinity.
        combined_dev = max(0.0, behavior_dev - 1.0) + temporal_dev
        
        t_adaptive = t_base / (1.0 + self.k_sensitivity * combined_dev)
        return float(t_adaptive)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "k_sensitivity": self.k_sensitivity,
            "global_threshold": self.global_threshold,
            "host_thresholds": self.host_thresholds,
            "is_fitted": self.is_fitted
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        state = joblib.load(filepath)
        obj = cls(k_sensitivity=state["k_sensitivity"])
        obj.global_threshold = state["global_threshold"]
        obj.host_thresholds = state["host_thresholds"]
        obj.is_fitted = state["is_fitted"]
        return obj
