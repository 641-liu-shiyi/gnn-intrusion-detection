import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    root / "data" / "processed" / "train.csv",
    nrows=10000
)

features = np.load(
    root / "data" / "features" / "X_train.npy",
    mmap_mode="r"
)[:10000]

ips = pd.unique(
    pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]])
)

node_map = {ip: i for i, ip in enumerate(ips)}

src = df["IPV4_SRC_ADDR"].map(node_map).to_numpy()
dst = df["IPV4_DST_ADDR"].map(node_map).to_numpy()

edge_index = np.vstack([src, dst])
edge_attr = np.asarray(features)
edge_label = df["Label"].to_numpy()

print("Nodes:", len(node_map))
print("Edges:", edge_index.shape[1])
print("Edge index:", edge_index.shape)
print("Edge features:", edge_attr.shape)
print("Edge labels:", edge_label.shape)

print("\nFirst 5 edges:")

for i in range(5):
    print(
        edge_index[0, i],
        "->",
        edge_index[1, i],
        "|",
        df.iloc[i]["IPV4_SRC_ADDR"],
        "->",
        df.iloc[i]["IPV4_DST_ADDR"],
        "| Label:",
        edge_label[i]
    )