import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proposed.preprocessing.transformer import SIEMPreprocessor
from proposed.behavioral.host_baseline import HostBehaviorBaseline
from proposed.behavioral.temporal_baseline import TemporalBaseline
from proposed.thresholding.adaptive import AdaptiveThresholding
from proposed.scoring.risk import RiskScorer
from baseline.mini_siem.model import Autoencoder

st.set_page_config(page_title="Adaptive Behavioral SIEM", layout="wide")
st.title("🔐 Adaptive Behavioral SIEM")

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(root, "results", "models")

@st.cache_resource
def load_models():
    preprocessor = SIEMPreprocessor.load(os.path.join(model_dir, "preprocessor.pkl"))
    host_model = HostBehaviorBaseline.load(os.path.join(model_dir, "host_baseline.pkl"))
    temp_model = TemporalBaseline.load(os.path.join(model_dir, "temporal_baseline.pkl"))
    threshold_model = AdaptiveThresholding.load(os.path.join(model_dir, "adaptive_threshold.pkl"))
    
    ml_model = Autoencoder()
    state = torch.load(os.path.join(root, "models", "autoencoder_beth.pth"), map_location="cpu")
    ml_model.load_state_dict(state)
    ml_model.eval()
    
    return preprocessor, host_model, temp_model, threshold_model, ml_model

preprocessor, host_model, temp_model, threshold_model, ml_model = load_models()
scorer = RiskScorer()

uploaded_file = st.file_uploader("Upload Security Log CSV (BETH format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Stream for UI (first 1000 for speed)
    df = df.sort_values(by="timestamp").head(2000).reset_index(drop=True)
    
    with st.spinner("Processing Replay Stream..."):
        out = preprocessor.transform(df)
        X_test = torch.tensor(out["X_scaled"], dtype=torch.float32)
        hosts = out["hosts"]
        timestamps = out["timestamps"]
        
        with torch.no_grad():
            recon = ml_model(X_test)
            ml_errors = torch.mean((X_test - recon) ** 2, dim=1).numpy()
            
        results = []
        host_events = {h: [] for h in set(hosts)}
        window_s = host_model.frequency_window_s
        
        for i in range(len(ml_errors)):
            h = hosts[i]
            t = timestamps[i]
            q = host_events[h]
            q.append(t)
            while len(q) > 0 and q[0] < t - window_s:
                q.pop(0)
                
            freq = len(q)
            beh_dev = host_model.calculate_deviation(h, freq)
            temp_dev = temp_model.calculate_deviation(t)
            
            adap_thresh = threshold_model.calculate_threshold(h, beh_dev, temp_dev)
            risk_score = scorer.calculate_risk(ml_errors[i], beh_dev, temp_dev, adap_thresh)
            alert_level = scorer.classify_alert(risk_score, ml_errors[i], adap_thresh)
            
            results.append({
                "Timestamp": t,
                "Host": h,
                "Event Frequency": freq,
                "ML Score": float(ml_errors[i]),
                "Behavior Dev": float(beh_dev),
                "Adaptive Threshold": float(adap_thresh),
                "Risk Score": float(risk_score),
                "Alert Level": alert_level
            })
            
    results_df = pd.DataFrame(results)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(results_df))
    col2.metric("Anomalies (ML > Thresh)", len(results_df[results_df["ML Score"] > results_df["Adaptive Threshold"]]))
    col3.metric("High/Critical Risk", len(results_df[results_df["Alert Level"].isin(["HIGH", "CRITICAL"])]))
    
    st.subheader("Priority Alerts")
    alerts = results_df[results_df["Alert Level"].isin(["HIGH", "CRITICAL"])].sort_values("Risk Score", ascending=False)
    st.dataframe(alerts, use_container_width=True)
    
    st.subheader("Event Investigation")
    if len(alerts) > 0:
        event = alerts.iloc[0]
        st.markdown(f"### Investigating High-Risk Event on **{event['Host']}**")
        st.markdown(f"""
        **Risk Score:** {event['Risk Score']:.2f}  
        **ML Score:** {event['ML Score']:.2f} (Threshold: {event['Adaptive Threshold']:.2f})  
        **Behavior Deviation:** {event['Behavior Dev']:.2f}
        
        *Reason:* The system flagged this because the risk score combines an ML reconstruction anomaly with significant behavioral deviation from the host baseline. 
        """)
    else:
        st.info("No high priority alerts to investigate.")
