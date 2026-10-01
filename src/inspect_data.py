import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "NF-UNSW-NB15-v2.csv"

df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nDuplicate rows:", df.duplicated().sum())

print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nAttack distribution:")
print(df["Attack"].value_counts())