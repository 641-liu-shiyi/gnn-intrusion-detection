import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    root / "data" / "processed" / "train.csv",
    nrows=100000
)

window_sizes = [1000, 5000, 10000]

for window_size in window_sizes:
    node_counts = []
    malicious_counts = []

    for start in range(0, len(df), window_size):
        window = df.iloc[start:start + window_size]

        if len(window) < window_size:
            continue

        ips = pd.unique(
            pd.concat([
                window["IPV4_SRC_ADDR"],
                window["IPV4_DST_ADDR"]
            ])
        )

        node_counts.append(len(ips))
        malicious_counts.append(int(window["Label"].sum()))

    print(f"\nWindow size: {window_size}")
    print("Graphs:", len(node_counts))
    print("Average nodes:", round(sum(node_counts) / len(node_counts), 2))
    print("Min nodes:", min(node_counts))
    print("Max nodes:", max(node_counts))
    print("Average malicious edges:", round(sum(malicious_counts) / len(malicious_counts), 2))
    print("Graphs with malicious edges:", sum(x > 0 for x in malicious_counts))