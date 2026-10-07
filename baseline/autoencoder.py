import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# LOAD DATA
# -------------------------

X_train = pd.read_csv("processed_dataset/X_train.csv").values
y_train = pd.read_csv("processed_dataset/y_train.csv").values.squeeze()

X_val = pd.read_csv("processed_dataset/X_val.csv").values
y_val = pd.read_csv("processed_dataset/y_val.csv").values.squeeze()

X_test = pd.read_csv("processed_dataset/X_test.csv").values
y_test = pd.read_csv("processed_dataset/y_test.csv").values.squeeze()

# -------------------------
# TRAIN ONLY NORMAL DATA
# -------------------------

X_train_normal = X_train[y_train == 0]

print("Training samples used:", X_train_normal.shape)

# -------------------------
# PYTORCH DATASET
# -------------------------

train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
train_loader = DataLoader(train_tensor, batch_size=512, shuffle=True)

# -------------------------
# AUTOENCODER MODEL
# -------------------------

input_dim = X_train.shape[1]

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# TRAIN MODEL
# -------------------------

epochs = 15
loss_history = []

for epoch in range(epochs):

    epoch_loss = 0

    for batch in train_loader:

        output = model(batch)

        loss = criterion(output, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)

    loss_history.append(epoch_loss)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

# -------------------------
# LOSS CURVE
# -------------------------

plt.plot(loss_history)
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# -------------------------
# RECONSTRUCTION ERROR
# -------------------------

def reconstruction_error(data):

    tensor = torch.tensor(data, dtype=torch.float32)

    with torch.no_grad():
        recon = model(tensor)

    mse = ((tensor - recon) ** 2).mean(axis=1)

    return mse.numpy()

# -------------------------
# VALIDATION
# -------------------------

val_error = reconstruction_error(X_val)

threshold = np.percentile(val_error, 98)

val_pred = (val_error > threshold).astype(int)

print("\nValidation Results")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred))

# -------------------------
# TEST
# -------------------------

test_error = reconstruction_error(X_test)

test_pred = (test_error > threshold).astype(int)

print("\nTest Results")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))

# -------------------------
# SAVE MODEL
# -------------------------

torch.save(model.state_dict(), "models/autoencoder_beth.pth")

print("\nModel saved successfully")