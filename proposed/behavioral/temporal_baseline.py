import pandas as pd
import numpy as np
import os
import joblib

class TemporalBaseline:
    def __init__(self):
        # Maps hour of day (0-23) to probability/frequency of events
        self.hourly_distribution = np.zeros(24)
        self.is_fitted = False
        self.total_events = 0

    def fit(self, df: pd.DataFrame):
        if "timestamp" not in df.columns:
            raise KeyError("DataFrame must contain 'timestamp'")

        # In BETH, timestamps might be UNIX epochs (seconds).
        # We will extract the hour of the day.
        # Ensure we drop NaNs.
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        if len(ts) == 0:
            self.is_fitted = True
            return

        # BETH timestamp might be in float seconds or ms. If it's a huge number, it's ms.
        # Assuming seconds for standard UNIX timestamp.
        dt = pd.to_datetime(ts, unit='s', errors='coerce')
        # Fallback if out of bounds (might be ms):
        if dt.isna().any():
            dt = pd.to_datetime(ts, unit='ms', errors='coerce')
            
        hours = dt.dt.hour.dropna().values
        
        counts, _ = np.histogram(hours, bins=np.arange(25))
        self.total_events = np.sum(counts)
        
        if self.total_events > 0:
            self.hourly_distribution = counts / self.total_events
        else:
            self.hourly_distribution = np.ones(24) / 24.0
            
        self.is_fitted = True

    def calculate_deviation(self, timestamp: float) -> float:
        """
        Calculates how unusual an event is temporally.
        High deviation means the event occurred at an unusually quiet hour.
        Deviation = 1.0 - probability_at_hour
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
            
        try:
            dt = pd.to_datetime([timestamp], unit='s')
            if dt.isna().any():
                dt = pd.to_datetime([timestamp], unit='ms')
            hour = dt.dt.hour.values[0]
        except Exception:
            # Fallback for completely invalid timestamps
            hour = 12

        prob = self.hourly_distribution[hour]
        # Max prob is 1.0. If prob is very small, deviation is close to 1.
        # We normalize it against the most active hour.
        max_prob = np.max(self.hourly_distribution)
        if max_prob <= 0:
            return 0.0
            
        rel_prob = prob / max_prob
        return float(1.0 - rel_prob)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "hourly_distribution": self.hourly_distribution,
            "total_events": self.total_events,
            "is_fitted": self.is_fitted
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        state = joblib.load(filepath)
        obj = cls()
        obj.hourly_distribution = state["hourly_distribution"]
        obj.total_events = state["total_events"]
        obj.is_fitted = state["is_fitted"]
        return obj
