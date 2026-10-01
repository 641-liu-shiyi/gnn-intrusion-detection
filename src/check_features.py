import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "processed" / "train.csv"

df = pd.read_csv(path)

features = df.drop(columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"])

print("Number of features:", len(features.columns))
print("\nData types:")
print(features.dtypes.value_counts())

print("\nFeature ranges:")
for col in features.columns:
    print(
        col,
        "min =", features[col].min(),
        "max =", features[col].max(),
        "mean =", round(features[col].mean(), 2)
    )