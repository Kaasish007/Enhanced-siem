import os
import sys
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proposed.preprocessing.transformer import SIEMPreprocessor
from proposed.behavioral.host_baseline import HostBehaviorBaseline
from proposed.behavioral.temporal_baseline import TemporalBaseline
from proposed.thresholding.adaptive import AdaptiveThresholding
from baseline.mini_siem.model import Autoencoder

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "results", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(root, "BETH", "labelled_training_data.csv"))
    val_df = pd.read_csv(os.path.join(root, "BETH", "labelled_validation_data.csv"))
    
    print("Fitting preprocessor...")
    # Preprocessor fits on normal training data (or whole training?)
    # Original baseline fits StandardScaler on all train/test incrementally? No, we fit strictly on train
    preprocessor = SIEMPreprocessor()
    preprocessor.fit(train_df)
    preprocessor.save(os.path.join(model_dir, "preprocessor.pkl"))
    
    print("Fitting Host Baseline...")
    host_model = HostBehaviorBaseline(frequency_window_s=3600) # 1-hour windows
    host_model.fit(train_df)
    host_model.save(os.path.join(model_dir, "host_baseline.pkl"))
    
    print("Fitting Temporal Baseline...")
    temp_model = TemporalBaseline()
    temp_model.fit(train_df)
    temp_model.save(os.path.join(model_dir, "temporal_baseline.pkl"))
    
    print("Generating Validation Errors...")
    # Load ML Model
    val_out = preprocessor.transform(val_df)
    X_val = torch.tensor(val_out["X_scaled"], dtype=torch.float32)
    hosts = val_out["hosts"]
    
    # We use validation data reconstruction errors to determine threshold parameters
    # The Baseline model is saved at root/models/autoencoder_beth.pth
    ml_model = Autoencoder()
    state = torch.load(os.path.join(root, "models", "autoencoder_beth.pth"), map_location="cpu")
    ml_model.load_state_dict(state)
    ml_model.eval()
    
    with torch.no_grad():
        recon = ml_model(X_val)
        errors = torch.mean((X_val - recon) ** 2, dim=1).numpy()
        
    print("Fitting Adaptive Threshold...")
    threshold_model = AdaptiveThresholding(k_sensitivity=0.5)
    threshold_model.fit(hosts, errors, base_pct=98.0)
    threshold_model.save(os.path.join(model_dir, "adaptive_threshold.pkl"))
    
    print("Training pipeline complete.")

if __name__ == "__main__":
    main()
