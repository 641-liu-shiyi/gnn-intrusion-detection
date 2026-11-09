"""Phase 1: read-only check of Member 1's outputs. Writes nothing.

Usage: python member3/phase1_verify_inputs.py <repo_root>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
SUSPECT = ("ADDR", "TTL", "PORT", "FLOW_START", "FLOW_END", "DNS_QUERY_ID")

print("class_names.json files found:")
for p in sorted(root.glob("data/**/class_names.json")):
    print(f"  {p.relative_to(root)}: {json.loads(p.read_text())}")

datasets = {"UNSW": ("features", "graphs_unsw"), "CICIDS": ("features_cicids", "graphs_cicids")}

for name, (feat_dir, graph_dir) in datasets.items():
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    # 1. Feature names: count and identifier-like columns
    fn_path = root / "data" / feat_dir / "feature_names.joblib"
    if fn_path.exists():
        names = joblib.load(fn_path)
        print("feature count:", len(names))
        print("suspect features:", [n for n in names if any(s in n for s in SUSPECT)])
        scaler = joblib.load(root / "data" / feat_dir / "scaler.joblib")
        print("scaler mean[:5]:", np.round(scaler.mean_[:5], 3))
    else:
        print("MISSING:", fn_path)

    # 2. Graph files: count, keys, shapes, both label distributions per split
    gdir = root / "data" / graph_dir
    for split in ["train", "val", "test"]:
        files = sorted(gdir.glob(f"{split}_graph_*.npz"))
        if not files:
            print(f"{split}: NO GRAPH FILES in {gdir}")
            continue
        with np.load(files[0]) as g:
            keys, shape = g.files, g["edge_attr"].shape
        labels = {k: Counter() for k in ("edge_label", "edge_attack") if k in keys}
        edges = nodes = 0
        for f in files:
            with np.load(f) as gi:
                for k in labels:
                    labels[k].update(gi[k].tolist())
                edges += len(gi["edge_attr"])
                nodes += int(gi["num_nodes"])
        print(f"\n{split}: {len(files)} graphs | keys={keys} | edge_attr={shape} | "
              f"edges={edges:,} | avg nodes/graph={nodes / len(files):.1f}")
        for k, c in labels.items():
            print(f"  {k} ({len(c)} values): {dict(sorted(c.items()))}")
