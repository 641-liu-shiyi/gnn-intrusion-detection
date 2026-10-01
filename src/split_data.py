import pandas as pd
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "NF-UNSW-NB15-v2.csv"
output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

train.to_csv(output_dir / "train.csv", index=False)
val.to_csv(output_dir / "val.csv", index=False)
test.to_csv(output_dir / "test.csv", index=False)

print("Total:", len(df))
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print("\nTrain labels:")
print(train["Label"].value_counts())

print("\nValidation labels:")
print(val["Label"].value_counts())

print("\nTest labels:")
print(test["Label"].value_counts())