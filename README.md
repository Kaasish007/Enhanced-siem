# Mini SIEM

Mini SIEM is a small Streamlit application that uploads a **CSV** of log events and flags anomalies using a trained **autoencoder** model.

## Requirements

Install Python 3.10+.

Then install dependencies:

```powershell
cd "g:\incognito\incognito"
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit --version
```

PyTorch is installed as **CPU-only** to make hosting/CI easier.

## Run

```powershell
cd "g:\incognito\incognito"
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

## Input (CSV)

You upload a CSV in the app UI.

Important details:
- The model uses **14 features** (`INPUT_DIM = 14`).
- Columns are preprocessed automatically:
  - Drops label columns if present: `sus`, `evil`, `attack`, `label`, `anomaly`
  - If columns `args` / `stackAddresses` exist, they are converted to string length
  - Object/categorical columns are converted to numeric codes
  - Everything is converted to numeric; invalid values become `0`
- If more than 14 columns remain after preprocessing, the app uses the **first 14 columns in order**.
- If fewer than 14 remain, it **pads with zeros**.

If your results look wrong, the most likely cause is **feature column order** in your CSV.

## Deploy to Streamlit Community Cloud (free/cheap)

This is the simplest “faculty demo via URL” option.


1. Create a GitHub repo and push your project files (make sure `app.py`, `requirements.txt`, `mini_siem/`, and `models/` are included).
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in.
3. Click **New app**.
4. Choose your GitHub repo (the branch is usually `main`).
5. Set the **App file** to `app.py` and deploy.
6. After a few minutes you’ll get a shareable URL. Faculty only need to open that link and upload the CSV.

Notes for reliability:
- The autoencoder model file is small (`models/autoencoder_beth.pth`), so it should bundle fine.
- If `model_results.csv` is missing, the app will just show a warning in Tab 2 (Tab 1 still works).

