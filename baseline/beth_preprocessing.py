import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# PATHS
# -----------------------------

folder = "BETH"
output_folder = "processed_dataset"
os.makedirs(output_folder, exist_ok=True)

train_path = os.path.join(folder, "labelled_training_data.csv")
val_path = os.path.join(folder, "labelled_validation_data.csv")
test_path = os.path.join(folder, "labelled_testing_data.csv")

# -----------------------------
# LOAD DATA
# -----------------------------

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------

def preprocess(df):

    df = df.copy()

    # Convert timestamp
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Numeric columns
    numeric_cols = [
        "processId",
        "threadId",
        "parentProcessId",
        "userId",
        "mountNamespace",
        "eventId",
        "argsNum",
        "returnValue"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric nulls
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Categorical columns
    categorical_cols = [
        "processName",
        "hostName",
        "eventName",
        "stackAddresses",
        "args"
    ]

    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # Create attack label
    df["sus"] = pd.to_numeric(df["sus"], errors="coerce").fillna(0)
    df["evil"] = pd.to_numeric(df["evil"], errors="coerce").fillna(0)

    df["attack"] = (df["sus"] + df["evil"]).apply(lambda x: 1 if x > 0 else 0)

    df.drop(columns=["sus","evil"], inplace=True)

    return df


train_df = preprocess(train_df)
val_df = preprocess(val_df)
test_df = preprocess(test_df)

# -----------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------

categorical_cols = [
    "processName",
    "hostName",
    "eventName",
    "stackAddresses",
    "args"
]

encoders = {}

for col in categorical_cols:

    encoder = LabelEncoder()

    combined = pd.concat([
        train_df[col].astype(str),
        val_df[col].astype(str),
        test_df[col].astype(str)
    ])

    encoder.fit(combined)

    train_df[col] = encoder.transform(train_df[col].astype(str))
    val_df[col] = encoder.transform(val_df[col].astype(str))
    test_df[col] = encoder.transform(test_df[col].astype(str))

    encoders[col] = encoder


# -----------------------------
# FEATURE / LABEL SPLIT
# -----------------------------

X_train = train_df.drop("attack", axis=1)
y_train = train_df["attack"]

X_val = val_df.drop("attack", axis=1)
y_val = val_df["attack"]

X_test = test_df.drop("attack", axis=1)
y_test = test_df["attack"]

# -----------------------------
# FEATURE SCALING
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -----------------------------
# SAVE DATASETS
# -----------------------------

pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
pd.DataFrame(X_val).to_csv(f"{output_folder}/X_val.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)

y_train.to_csv(f"{output_folder}/y_train.csv", index=False)
y_val.to_csv(f"{output_folder}/y_val.csv", index=False)
y_test.to_csv(f"{output_folder}/y_test.csv", index=False)

print("Preprocessing complete.")
print("Datasets saved in:", output_folder)