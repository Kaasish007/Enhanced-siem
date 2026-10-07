import numpy as np

class RiskScorer:
    def __init__(self, w1=0.5, w2=0.35, w3=0.15):
        self.w1 = w1 # ML Anomaly Weight
        self.w2 = w2 # Behavioral Dev Weight
        self.w3 = w3 # Temporal Dev Weight
        
    def _sigmoid_norm(self, x, scale=1.0, offset=0.0):
        # Maps [0, inf) to [0, 1) smoothly
        return 1.0 / (1.0 + np.exp(-(x - offset) / scale))

    def calculate_risk(self, ml_score: float, behavior_dev: float, temporal_dev: float, adaptive_threshold: float) -> float:
        """
        Risk Score = w1 * Norm(ML_Score) + w2 * Norm(Behavior_Dev) + w3 * Norm(Temporal_Dev)
        """
        # We normalize ML score such that ML = adaptive_threshold maps to ~0.5
        # If ML score >> threshold, norm -> 1.0
        ml_norm = self._sigmoid_norm(ml_score, scale=adaptive_threshold*0.5, offset=adaptive_threshold)
        
        # Behavior dev normally around 1.0 for regular, >3.0 is anomalous. 
        # Map 1.0 -> ~0.1, 3.0 -> ~0.5, 6.0 -> ~0.9
        beh_norm = self._sigmoid_norm(behavior_dev, scale=1.0, offset=3.0)
        
        # Temporal dev is already [0, 1.0].
        temp_norm = temporal_dev
        
        risk = (self.w1 * ml_norm) + (self.w2 * beh_norm) + (self.w3 * temp_norm)
        return float(risk)
        
    def classify_alert(self, risk_score: float, ml_score: float, adaptive_threshold: float):
        ml_flags = ml_score > adaptive_threshold
        if risk_score > 0.8 or (ml_flags and risk_score > 0.6):
            return "CRITICAL"
        elif risk_score > 0.6 or ml_flags:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        return "LOW"
