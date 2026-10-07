import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


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

print("Training samples used:", X_train_normal.shape)


# -------------------------
# TRAIN MODEL
# -------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.002,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_normal)


# -------------------------
# VALIDATION TEST
# -------------------------

val_scores = model.decision_function(X_val)

threshold = val_scores.quantile(0.02)

val_pred = (val_scores < threshold).astype(int)

print("\nValidation Results")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred))


# -------------------------
# TEST SET
# -------------------------

test_scores = model.decision_function(X_test)

test_pred = (test_scores < threshold).astype(int)

print("\nTest Results")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))