import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Mini SIEM", layout="wide")
st.title("🔐 Mini SIEM Log Analysis & Threat Detection")

INPUT_DIM = 14
MODEL_PATH = "models/autoencoder_beth.pth"

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,INPUT_DIM)
        )
    def forward(self,x):
        return self.decoder(self.encoder(x))

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

def preprocess_logs(df):
    df = df.copy()
    label_cols = ["sus", "evil", "attack", "label", "anomaly"]
    df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")
    if "args" in df.columns:
        df["args"] = df["args"].astype(str).str.len()
    if "stackAddresses" in df.columns:
        df["stackAddresses"] = df["stackAddresses"].astype(str).str.len()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

def run_inference(df):
    features = preprocess_logs(df)
    if features.shape[1] > INPUT_DIM:
        features = features.iloc[:, :INPUT_DIM]
    elif features.shape[1] < INPUT_DIM:
        missing = INPUT_DIM - features.shape[1]
        for i in range(missing):
            features[f"pad_{i}"] = 0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values.astype(np.float32))
    X_tensor = torch.tensor(X_scaled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_tensor = X_tensor.to(device)
    with torch.no_grad():
        recon = model(X_tensor)
        error = torch.mean((X_tensor - recon)**2, dim=1).cpu().numpy()
    threshold = np.percentile(error, 98.0)
    # Add new columns directly to the original dataframe
    df["Status"] = ["Threat" if e > threshold else "Normal" for e in error]
    df["Result"] = df["Status"].map({
        "Threat": "🔴 Threat",
        "Normal": "🟢 Normal"
    })
    return df

uploaded_file = st.file_uploader("Upload Security Log CSV")

if uploaded_file and model is not None:
    df = pd.read_csv(uploaded_file)
    results = run_inference(df)

    # Filter selector based on the new Result column
    choice = st.radio("Filter by Result:", ["All", "🔴 Threat", "🟢 Normal"])
    if choice != "All":
        results = results[results["Result"] == choice]

    # --- Threat/Normal counts ---
    total_threats = (results["Status"] == "Threat").sum()
    total_normals = (results["Status"] == "Normal").sum()
    total_events = len(results)

    st.subheader("Summary Counts")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", total_events)
    col2.metric("Threats Detected", total_threats)
    col3.metric("Normal Events", total_normals)

    # --- Display full dataset with new Result column ---
    st.subheader("Detection Results")
    st.dataframe(results, hide_index=True)

    st.download_button(
        label="Download Results",
        data=results.to_csv(index=False),
        file_name="threat_results.csv",
        mime="text/csv"
    )
elif not uploaded_file:
    st.info("Upload a CSV file to begin analysis.")
else:
    st.warning("Model not loaded. Please check model path.")
