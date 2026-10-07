import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proposed.preprocessing.transformer import SIEMPreprocessor
from proposed.behavioral.host_baseline import HostBehaviorBaseline
from proposed.behavioral.temporal_baseline import TemporalBaseline
from proposed.thresholding.adaptive import AdaptiveThresholding
from proposed.scoring.risk import RiskScorer
from baseline.mini_siem.model import Autoencoder

def evaluate_models():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "results", "models")
    
    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(root, "BETH", "labelled_testing_data.csv"))
    
    print("Loading models...")
    preprocessor = SIEMPreprocessor.load(os.path.join(model_dir, "preprocessor.pkl"))
    host_model = HostBehaviorBaseline.load(os.path.join(model_dir, "host_baseline.pkl"))
    temp_model = TemporalBaseline.load(os.path.join(model_dir, "temporal_baseline.pkl"))
    threshold_model = AdaptiveThresholding.load(os.path.join(model_dir, "adaptive_threshold.pkl"))
    
    ml_model = Autoencoder()
    state = torch.load(os.path.join(root, "models", "autoencoder_beth.pth"), map_location="cpu")
    ml_model.load_state_dict(state)
    ml_model.eval()
    
    print("Preprocessing Test Set...")
    # Get labels
    test_clean, y_true = preprocessor.extract_labels(test_df)
    
    # We will simulate stream for behavioral frequency
    test_clean = test_clean.sort_values(by="timestamp").reset_index(drop=True)
    y_true = y_true.loc[test_clean.index].values
    
    out = preprocessor.transform(test_clean)
    X_test = torch.tensor(out["X_scaled"], dtype=torch.float32)
    hosts = out["hosts"]
    timestamps = out["timestamps"]
    
    print("Computing ML Anomaly Scores...")
    with torch.no_grad():
        recon = ml_model(X_test)
        ml_errors = torch.mean((X_test - recon) ** 2, dim=1).numpy()
        
    print("Streaming Simulation: Calculating Behavioral Features...")
    behavioral_devs = np.zeros(len(ml_errors))
    temporal_devs = np.zeros(len(ml_errors))
    
    # Fast window streaming using frequency dict for test set
    host_events = {h: [] for h in set(hosts)}
    window_s = host_model.frequency_window_s
    
    for i in range(len(ml_errors)):
        h = hosts[i]
        t = timestamps[i]
        
        # update recent events
        q = host_events[h]
        q.append(t)
        # trim old
        while len(q) > 0 and q[0] < t - window_s:
            q.pop(0)
            
        freq_current = len(q)
        behavioral_devs[i] = host_model.calculate_deviation(h, freq_current)
        temporal_devs[i] = temp_model.calculate_deviation(t)
        
    print("Evaluating Systems...")
    results = []
    
    # Model A: Baseline - Global dynamic threshold (original app behavior)
    batch_threshold = float(np.percentile(ml_errors, 98.0))
    y_pred_A = (ml_errors > batch_threshold).astype(int)
    
    # Model E: Adaptive Threshold System
    y_pred_E = np.zeros(len(ml_errors))
    adaptive_thresholds = np.zeros(len(ml_errors))
    for i in range(len(ml_errors)):
        t_adap = threshold_model.calculate_threshold(hosts[i], behavioral_devs[i], temporal_devs[i])
        adaptive_thresholds[i] = t_adap
        if ml_errors[i] > t_adap:
            y_pred_E[i] = 1
            
    # Model F: Risk Scoring
    scorer = RiskScorer()
    y_pred_F = np.zeros(len(ml_errors))
    risk_scores = np.zeros(len(ml_errors))
    for i in range(len(ml_errors)):
        r = scorer.calculate_risk(ml_errors[i], behavioral_devs[i], temporal_devs[i], adaptive_thresholds[i])
        risk_scores[i] = r
        # Let's say risk > 0.6 is alert
        if r > 0.6:
            y_pred_F[i] = 1
            
    def get_metrics(name, pred):
        return {
            "Model": name,
            "F1": f1_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "Precision": precision_score(y_true, pred, zero_division=0)
        }
        
    results.append(get_metrics("Baseline (Global Dynamic)", y_pred_A))
    results.append(get_metrics("Proposed (Adaptive Threshold)", y_pred_E))
    results.append(get_metrics("Proposed (Full Risk Score > 0.6)", y_pred_F))
    
    df_results = pd.DataFrame(results)
    print("\n--- ABLATION RESULTS ---")
    print(df_results)
    
    df_results.to_csv(os.path.join(root, "results", "ablation.csv"), index=False)
    print("\nSaved to results/ablation.csv")
    
if __name__ == "__main__":
    evaluate_models()
