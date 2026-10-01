import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "NF-CSE-CIC-IDS2018-v2.csv"

seen = set()
duplicates = 0
total = 0

for chunk in pd.read_csv(path, chunksize=200000):
    hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()

    for h in hashes:
        h = int(h)
        if h in seen:
            duplicates += 1
        else:
            seen.add(h)

    total += len(chunk)
    print(f"Processed: {total:,} | Duplicates: {duplicates:,}")

print("\nTotal:", total)
print("Duplicates:", duplicates)
print("Unique:", total - duplicates)