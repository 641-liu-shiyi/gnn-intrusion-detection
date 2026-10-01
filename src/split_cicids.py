import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
input_path = root / "data" / "NF-CSE-CIC-IDS2018-v2.csv"
output_dir = root / "data" / "processed_cicids"
output_dir.mkdir(parents=True, exist_ok=True)

total_unique = 18893389
train_end = int(total_unique * 0.70)
val_end = int(total_unique * 0.80)

seen = set()
written = 0

train_path = output_dir / "train.csv"
val_path = output_dir / "val.csv"
test_path = output_dir / "test.csv"

for p in [train_path, val_path, test_path]:
    if p.exists():
        p.unlink()

for chunk in pd.read_csv(input_path, chunksize=200000):
    hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()

    keep = []
    for h in hashes:
        h = int(h)
        if h in seen:
            keep.append(False)
        else:
            seen.add(h)
            keep.append(True)

    chunk = chunk[keep]

    while len(chunk) > 0:
        if written < train_end:
            path = train_path
            limit = train_end
        elif written < val_end:
            path = val_path
            limit = val_end
        else:
            path = test_path
            limit = total_unique

        take = min(len(chunk), limit - written)
        part = chunk.iloc[:take]

        part.to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False
        )

        written += take
        chunk = chunk.iloc[take:]

    print(f"Written: {written:,}")

print("\nTrain:", train_end)
print("Validation:", val_end - train_end)
print("Test:", total_unique - val_end)
print("Total:", written)