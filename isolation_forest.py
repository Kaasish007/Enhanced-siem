import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold


# ---------------------------------
# LOAD DATA
# ---------------------------------

X_train = pd.read_csv("processed_dataset/X_train.csv")
y_train = pd.read_csv("processed_dataset/y_train.csv").squeeze()

X_val = pd.read_csv("processed_dataset/X_val.csv")
y_val = pd.read_csv("processed_dataset/y_val.csv").squeeze()

X_test = pd.read_csv("processed_dataset/X_test.csv")
y_test = pd.read_csv("processed_dataset/y_test.csv").squeeze()


# ---------------------------------
# REMOVE ATTACKS FROM TRAINING
# ---------------------------------

normal_index = y_train[y_train == 0].index
X_train_normal = X_train.loc[normal_index]

print("Training samples used:", X_train_normal.shape)
print("Attacks removed from training:", (y_train == 1).sum())


# ---------------------------------
# HYPERPARAMETER GRID
# ---------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_samples": ["auto", 0.8],
    "contamination": [0.001, 0.002, 0.005],
    "max_features": [0.8, 1.0]
}

grid = list(ParameterGrid(param_grid))

best_model = None
best_score = -np.inf
best_params = None


# ---------------------------------
# CROSS VALIDATION USING VALIDATION SET
# ---------------------------------

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\nStarting Hyperparameter Search...")

for params in grid:

    model = IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        contamination=params["contamination"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_normal)

    scores = model.decision_function(X_val)

    precision, recall, _ = precision_recall_curve(y_val, -scores)

    pr_auc = auc(recall, precision)

    print("Params:", params, "PR-AUC:", pr_auc)

    if pr_auc > best_score:
        best_score = pr_auc
        best_model = model
        best_params = params


print("\nBest Parameters:", best_params)
print("Best PR-AUC:", best_score)


# ---------------------------------
# VALIDATION EVALUATION
# ---------------------------------

val_scores = best_model.decision_function(X_val)

threshold = np.percentile(val_scores, 2)

val_pred = (val_scores < threshold).astype(int)

print("\nValidation Results")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred))


# ---------------------------------
# TEST EVALUATION
# ---------------------------------

test_scores = best_model.decision_function(X_test)

test_pred = (test_scores < threshold).astype(int)

print("\nTest Results")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))


# ---------------------------------
# ROC CURVE
# ---------------------------------

fpr, tpr, _ = roc_curve(y_test, -test_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="ROC AUC = %.3f" % roc_auc)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ---------------------------------
# PRECISION RECALL CURVE
# ---------------------------------

precision, recall, _ = precision_recall_curve(y_test, -test_scores)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label="PR AUC = %.3f" % pr_auc)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.legend()
plt.show()


# ---------------------------------
# ANOMALY SCORE DISTRIBUTION
# ---------------------------------

plt.figure()

plt.hist(test_scores[y_test==0], bins=50, alpha=0.6, label="Benign")
plt.hist(test_scores[y_test==1], bins=50, alpha=0.6, label="Attack")

plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Isolation Forest Score Distribution")
plt.legend()

plt.show()


# ---------------------------------
# SAVE MODEL
# ---------------------------------

joblib.dump(best_model, "models/isolation_forest_beth.pkl")

print("\nModel saved successfully")