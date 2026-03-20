import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "metropt-3.csv")

df = pd.read_csv(file_path)

# drop useless column
df = df.drop(columns=["Unnamed: 0"])

# convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

print(df.head())
print(df.info())