import pandas as pd
import os

from src.feature_engineering import (
    create_early_failure_labels,
    create_window_features
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "metropt-3.csv")

# STEP 1 — load
df = pd.read_csv(file_path)

df = df.drop(columns=["Unnamed: 0"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Raw shape:", df.shape)

# STEP 2 — labeling
df = create_early_failure_labels(df)

print("Failure events:", df["failure_event"].sum())
print("Target counts:\n", df["target"].value_counts())

# STEP 3 — windowing
feature_df = create_window_features(df)

print("Feature dataset shape:", feature_df.shape)
print("Window target counts:\n", feature_df["target"].value_counts())
print(feature_df.head())