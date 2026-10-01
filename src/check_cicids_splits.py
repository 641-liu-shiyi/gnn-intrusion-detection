import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
data_dir = root / "data" / "processed_cicids"

for name in ["train", "val", "test"]:
    counts = {0: 0, 1: 0}
    total = 0

    for chunk in pd.read_csv(
        data_dir / f"{name}.csv",
        usecols=["Label"],
        chunksize=500000
    ):
        vc = chunk["Label"].value_counts()

        for label, count in vc.items():
            counts[int(label)] += int(count)

        total += len(chunk)

    print(f"\n{name.upper()}")
    print("Total:", total)
    print("Benign:", counts[0])
    print("Malicious:", counts[1])
    print("Malicious %:", round(counts[1] / total * 100, 2))