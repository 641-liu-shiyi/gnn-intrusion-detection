import json

import pandas as pd
import numpy as np
from pathlib import Path


root = Path(__file__).resolve().parent.parent

data_dir = root / "data" / "processed_cicids"
feature_dir = root / "data" / "features_cicids"
graph_dir = root / "data" / "graphs_cicids"

graph_dir.mkdir(parents=True, exist_ok=True)

window_size = 5000


# --------------------------------------------------
# Build one consistent Attack -> integer ID mapping
# across train / val / test
# --------------------------------------------------

attack_classes = set()

for split in ["train", "val", "test"]:
    for chunk in pd.read_csv(
        data_dir / f"{split}.csv",
        usecols=["Attack"],
        chunksize=100000
    ):
        attack_classes.update(
            chunk["Attack"].dropna().astype(str).unique()
        )

class_names = sorted(attack_classes)

attack_to_id = {
    attack_name: class_id
    for class_id, attack_name in enumerate(class_names)
}

with open(
    graph_dir / "class_names.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        class_names,
        f,
        indent=2
    )

print("Attack classes:")
for class_id, attack_name in enumerate(class_names):
    print(f"{class_id}: {attack_name}")

print(f"\nSaved class names to {graph_dir / 'class_names.json'}")


# --------------------------------------------------
# Build graphs
# --------------------------------------------------

for split in ["train", "val", "test"]:
    print(f"\nProcessing {split}...")

    features = np.load(
        feature_dir / f"X_{split}.npy",
        mmap_mode="r"
    )

    offset = 0
    graph_id = 0

    for chunk in pd.read_csv(
        data_dir / f"{split}.csv",
        chunksize=window_size
    ):
        if len(chunk) < window_size:
            print(
                f"Skipping final incomplete window: "
                f"{len(chunk)} flows"
            )
            break

        ips = pd.unique(
            pd.concat([
                chunk["IPV4_SRC_ADDR"],
                chunk["IPV4_DST_ADDR"]
            ])
        )

        node_map = {
            ip: i
            for i, ip in enumerate(ips)
        }

        src = (
            chunk["IPV4_SRC_ADDR"]
            .map(node_map)
            .to_numpy()
        )

        dst = (
            chunk["IPV4_DST_ADDR"]
            .map(node_map)
            .to_numpy()
        )

        edge_index = np.vstack(
            [src, dst]
        ).astype(np.int64)

        edge_attr = np.asarray(
            features[offset:offset + window_size]
        ).astype(np.float32)

        # Existing binary label
        edge_label = (
            chunk["Label"]
            .to_numpy(dtype=np.int64)
        )

        # New multi-class attack label
        edge_attack = (
            chunk["Attack"]
            .astype(str)
            .map(attack_to_id)
            .to_numpy(dtype=np.int64)
        )

        np.savez_compressed(
            graph_dir / f"{split}_graph_{graph_id:05d}.npz",
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label=edge_label,
            edge_attack=edge_attack,
            num_nodes=len(node_map)
        )

        offset += window_size
        graph_id += 1

        if graph_id % 100 == 0:
            print(
                f"{split}: {graph_id} graphs"
            )

    print(f"{split} graphs: {graph_id}")

print("\nDone.")