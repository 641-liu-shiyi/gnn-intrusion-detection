"""Audit physical graph datasets on disk: counts of graphs, flows, classes, and features."""

import json
from pathlib import Path
import numpy as np

def audit():
    results = {}
    datasets = [
        ("UNSW", Path("graph_data_bundle/data/graphs_unsw")),
        ("CICIDS", Path("graph_data_bundle/data/graphs_cicids")),
    ]

    for dname, droot in datasets:
        print(f"\n[INFO] Auditing physical dataset: {dname} at {droot}")
        cnames_file = droot / "class_names.json"
        cnames = json.loads(cnames_file.read_text(encoding="utf-8")) if cnames_file.exists() else []
        benign_id = cnames.index("Benign") if "Benign" in cnames else 0

        results[dname] = {
            "root": str(droot),
            "class_names": cnames,
            "benign_id": benign_id,
            "splits": {}
        }

        for split in ["train", "val", "test"]:
            pattern = f"{split}_graph_*.npz"
            files = sorted(list(droot.glob(pattern)))
            file_count = len(files)
            print(f"  Split '{split}': {file_count} graph files found ({pattern}). Aggregating flow counts...")

            total_flows = 0
            feature_dim = None
            class_counts = {c: 0 for c in cnames}

            for i, fpath in enumerate(files):
                data = np.load(fpath, allow_pickle=False)
                edge_attr = data["edge_attr"]
                if feature_dim is None:
                    feature_dim = int(edge_attr.shape[1])
                num_edges = int(edge_attr.shape[0])
                total_flows += num_edges

                edge_attack = data["edge_attack"]
                unique, counts = np.unique(edge_attack, return_counts=True)
                for u, cnt in zip(unique, counts):
                    if int(u) < len(cnames):
                        class_counts[cnames[int(u)]] += int(cnt)

                if (i + 1) % 500 == 0 or (i + 1) == file_count:
                    print(f"    Processed {i+1}/{file_count} graphs ({total_flows:,} flows so far)...")

            results[dname]["splits"][split] = {
                "graph_count": file_count,
                "total_flows": total_flows,
                "feature_dim": feature_dim,
                "class_counts": class_counts
            }

    out_file = Path("results/physical_dataset_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[INFO] Audit complete. Saved physical audit to {out_file}")

if __name__ == "__main__":
    audit()
