import pandas as pd
from pathlib import Path
from collections import Counter

path = Path(__file__).resolve().parent.parent / "data" / "NF-CSE-CIC-IDS2018-v2.csv"

total = 0
missing = None
labels = Counter()
attacks = Counter()
chunksize = 200000

for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
    total += len(chunk)

    chunk_missing = chunk.isnull().sum()
    missing = chunk_missing if missing is None else missing.add(chunk_missing)

    labels.update(chunk["Label"].value_counts().to_dict())
    attacks.update(chunk["Attack"].value_counts().to_dict())

    print(f"Processed: {total:,}")

print("\nTotal rows:", total)

print("\nMissing values:")
print(missing[missing > 0])

print("\nLabel distribution:")
for k, v in sorted(labels.items()):
    print(k, v)

print("\nAttack distribution:")
for k, v in attacks.most_common():
    print(k, v)