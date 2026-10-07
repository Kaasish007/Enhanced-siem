import pandas as pd

X_train = pd.read_csv("processed_dataset/X_train.csv")
y_train = pd.read_csv("processed_dataset/y_train.csv").squeeze()

print("Before SMOTE:")
print(y_train.value_counts())