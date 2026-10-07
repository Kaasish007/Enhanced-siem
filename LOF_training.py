import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc
)
import joblib
# -------------------------
# LOAD DATA
# -------------------------

X_train = pd.read_csv("processed_dataset/X_train.csv")
y_train = pd.read_csv("processed_dataset/y_train.csv").squeeze()

X_val = pd.read_csv("processed_dataset/X_val.csv")
y_val = pd.read_csv("processed_dataset/y_val.csv").squeeze()

X_test = pd.read_csv("processed_dataset/X_test.csv")
y_test = pd.read_csv("processed_dataset/y_test.csv").squeeze()


# -------------------------
# TRAIN ONLY ON NORMAL DATA
# -------------------------

X_train_normal = X_train[y_train == 0]

# LOF is slow → sample
X_train_normal = X_train_normal.sample(100000, random_state=42)

print("Training samples used:", X_train_normal.shape)


# -------------------------
# HYPERPARAMETER GRID
# -------------------------

param_grid = [
    {"n_neighbors":10, "contamination":0.002},
    {"n_neighbors":20, "contamination":0.002},
    {"n_neighbors":30, "contamination":0.002},
    {"n_neighbors":50, "contamination":0.002},
]

best_model = None
best_score = 0
best_params = None


print("\nStarting LOF hyperparameter search")


# -------------------------
# TUNING LOOP
# -------------------------

for params in param_grid:

    lof = LocalOutlierFactor(
        n_neighbors=params["n_neighbors"],
        contamination=params["contamination"],
        novelty=True
    )

    lof.fit(X_train_normal)

    val_scores = lof.decision_function(X_val)

    precision, recall, _ = precision_recall_curve(y_val, -val_scores)

    pr_auc = auc(recall, precision)

    print("Params:", params, "PR-AUC:", pr_auc)

    if pr_auc > best_score:
        best_score = pr_auc
        best_model = lof
        best_params = params


print("\nBest parameters:", best_params)
print("Best validation PR-AUC:", best_score)


# -------------------------
# THRESHOLD SELECTION
# -------------------------

val_scores = best_model.decision_function(X_val)

threshold = np.percentile(val_scores, 2)

val_pred = (val_scores < threshold).astype(int)


print("\nValidation Results")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred))


# -------------------------
# TEST SET
# -------------------------

test_scores = best_model.decision_function(X_test)

test_pred = (test_scores < threshold).astype(int)

print("\nTest Results")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))

# ---------------------------------
# SAVE MODEL
# ---------------------------------

joblib.dump(best_model, "models/lof_beth.pkl")

print("\nModel saved successfully")