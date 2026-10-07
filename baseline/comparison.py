import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

# -------------------------
# LOAD DATA
# -------------------------

X_test = pd.read_csv("processed_dataset/X_test.csv").values
y_test = pd.read_csv("processed_dataset/y_test.csv").squeeze().values


# -------------------------
# LOAD SKLEARN MODELS
# -------------------------

iso_model = joblib.load("models/isolation_forest_beth.pkl")
lof_model = joblib.load("models/lof_beth.pkl")


# -------------------------
# ISOLATION FOREST
# -------------------------

iso_scores = iso_model.decision_function(X_test)

iso_thresh = np.percentile(iso_scores, 2)

iso_pred = (iso_scores < iso_thresh).astype(int)


# -------------------------
# LOF
# -------------------------

lof_scores = lof_model.decision_function(X_test)

lof_thresh = np.percentile(lof_scores, 2)

lof_pred = (lof_scores < lof_thresh).astype(int)


# -------------------------
# AUTOENCODER MODEL
# -------------------------

input_dim = X_test.shape[1]

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim,64),
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
            nn.Linear(64,input_dim)
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder()

model.load_state_dict(torch.load("models/autoencoder_beth.pth"))

model.eval()


# -------------------------
# AUTOENCODER PREDICTION
# -------------------------

X_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():

    recon = model(X_tensor)

error = ((X_tensor - recon)**2).mean(axis=1).numpy()

ae_thresh = np.percentile(error, 98)

ae_pred = (error > ae_thresh).astype(int)


# -------------------------
# METRICS
# -------------------------

results = pd.DataFrame({

"Model":[
"Isolation Forest",
"Local Outlier Factor",
"Autoencoder"
],

"Precision":[
precision_score(y_test,iso_pred),
precision_score(y_test,lof_pred),
precision_score(y_test,ae_pred)
],

"Recall":[
recall_score(y_test,iso_pred),
recall_score(y_test,lof_pred),
recall_score(y_test,ae_pred)
],

"F1":[
f1_score(y_test,iso_pred),
f1_score(y_test,lof_pred),
f1_score(y_test,ae_pred)
]

})


print(results)

results.to_csv("model_results.csv", index=False)