import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any

class SIEMPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.is_fitted = False
        
        self.numeric_cols = [
            "processId", "threadId", "parentProcessId", "userId", 
            "mountNamespace", "eventId", "argsNum", "returnValue"
        ]
        self.categorical_cols = [
            "processName", "hostName", "eventName", 
            "stackAddresses", "args"
        ]

    def _clean_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
        
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0
            
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("unknown").str.lower()
            else:
                df[col] = "unknown"
                
        return df

    def extract_labels(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        if "sus" in df.columns and "evil" in df.columns:
            sus = pd.to_numeric(df["sus"], errors="coerce").fillna(0)
            evil = pd.to_numeric(df["evil"], errors="coerce").fillna(0)
            attack = ((sus + evil) > 0).astype(int)
            df.drop(columns=["sus", "evil", "attack", "label", "anomaly"], errors="ignore", inplace=True)
            return df, attack
        return df, pd.Series(np.zeros(len(df)))

    def fit(self, df: pd.DataFrame):
        df = self._clean_base(df)
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Adding "unknown" explicitly to handle unseen data
            combined = pd.Series(["unknown"] + df[col].tolist())
            le.fit(combined)
            self.encoders[col] = le
            df[col] = le.transform(df[col])
            
        # Fit scaler on purely numerical + encoded categorical, ignoring timestamp for standard Autoencoder features
        # Timestamp is used for behavioral features, not for auto-encoder typically! 
        # But we must decide strictly what goes to autoencoder.
        # Original baseline uses all 13 features (numeric + cat) + attack label (which is removed).
        
        features = df[["timestamp"] + self.numeric_cols + self.categorical_cols]
        self.scaler.fit(features)
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> dict:
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted.")
            
        df = self._clean_base(df)
        
        # Save timestamp explicitly for behavioral analysis
        timestamps = df["timestamp"].values
        hosts = df["hostName"].values  # Need raw hostnames for baseline mapping
        
        for col in self.categorical_cols:
            le = self.encoders[col]
            # Handle unknown labels
            mapping = {label: idx for idx, label in enumerate(le.classes_)}
            df[col] = df[col].map(lambda s: mapping.get(s, mapping["unknown"]))
            
        features = df[["timestamp"] + self.numeric_cols + self.categorical_cols]
        X_scaled = self.scaler.transform(features)
        
        return {
            "X_scaled": X_scaled,
            "timestamps": timestamps,
            "hosts": hosts,
            "df_features": features, # Unscaled features with encoded categoricals
            "df_raw": df # Full cleaned df
        }
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "scaler": self.scaler,
            "encoders": self.encoders,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "is_fitted": self.is_fitted
        }
        joblib.dump(state, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        state = joblib.load(filepath)
        obj = cls()
        obj.scaler = state["scaler"]
        obj.encoders = state["encoders"]
        obj.numeric_cols = state["numeric_cols"]
        obj.categorical_cols = state["categorical_cols"]
        obj.is_fitted = state["is_fitted"]
        return obj
