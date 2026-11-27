"""Audit physical graph datasets on disk: graph counts, flows, classes, features, scaler, and integrity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np


def audit_dataset(
    name: str,
    graph_dir: Path,
    feature_dir: Path,
    reference_feature_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Audit a graph + feature directory pair without modifying data."""
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Missing graph directory for {name}: {graph_dir}")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Missing feature directory for {name}: {feature_dir}")

    cnames_file = graph_dir / "class_names.json"
    if not cnames_file.is_file():
        raise FileNotFoundError(f"Missing class_names.json at {cnames_file}")
    cnames: List[str] = json.loads(cnames_file.read_text(encoding="utf-8"))
    if "Benign" not in cnames:
        raise ValueError(f"'Benign' class missing from {cnames_file}")
    benign_id = cnames.index("Benign")

    scaler_path = feature_dir / "scaler.joblib"
    fnames_path = feature_dir / "feature_names.joblib"
    if not scaler_path.is_file() or not fnames_path.is_file():
        raise FileNotFoundError(f"Missing scaler/feature_names in {feature_dir}")

    scaler = joblib.load(scaler_path)
    feature_names = list(joblib.load(fnames_path))
    scaler_mean = getattr(scaler, "mean_", None)
    scaler_scale = getattr(scaler, "scale_", None)
    scaler_valid = (
        getattr(scaler, "n_features_in_", None) == 39
        and scaler_mean is not None
        and scaler_scale is not None
        and bool(np.all(np.isfinite(scaler_mean)))
        and bool(np.all(np.isfinite(scaler_scale)))
        and bool(np.all(scaler_scale > 0))
    )

    info: Dict[str, Any] = {
        "graph_dir": str(graph_dir),
        "feature_dir": str(feature_dir),
        "class_count": len(cnames),
        "class_names": cnames,
        "benign_id": benign_id,
        "feature_count": len(feature_names),
        "feature_names_match_reference": (
            feature_names == reference_feature_names if reference_feature_names is not None else True
        ),
        "scaler_type": type(scaler).__name__,
        "scaler_n_features_in": int(getattr(scaler, "n_features_in_", -1)),
        "scaler_valid": scaler_valid,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        pattern = f"{split}_graph_*.npz"
        files = sorted(list(graph_dir.glob(pattern)))
        file_count = len(files)
        print(f"  [{name}] Split '{split}': {file_count} graph files ({pattern})...")

        total_flows = 0
        feature_dim = None
        nan_or_inf_count = 0
        binary_mismatch_graphs = 0
        class_counts = {c: 0 for c in cnames}

        for i, fpath in enumerate(files):
            with np.load(fpath, allow_pickle=False) as data:
                edge_attr = data["edge_attr"]
                edge_attack = data["edge_attack"]
                edge_label = data["edge_label"]

            if feature_dim is None:
                feature_dim = int(edge_attr.shape[1])
            elif int(edge_attr.shape[1]) != feature_dim:
                raise ValueError(f"{fpath}: inconsistent feature dim {edge_attr.shape[1]} != {feature_dim}")

            if not np.all(np.isfinite(edge_attr)):
                nan_or_inf_count += int(np.sum(~np.isfinite(edge_attr)))

            derived_binary = (edge_attack != benign_id).astype(np.int64)
            if not np.array_equal(derived_binary, edge_label):
                binary_mismatch_graphs += 1

            total_flows += int(edge_attr.shape[0])
            unique, counts = np.unique(edge_attack, return_counts=True)
            for u, cnt in zip(unique, counts):
                idx = int(u)
                if 0 <= idx < len(cnames):
                    class_counts[cnames[idx]] += int(cnt)
                else:
                    raise ValueError(f"{fpath}: out-of-range class id {idx}")

            if (i + 1) % 500 == 0 or (i + 1) == file_count:
                print(f"    Processed {i+1}/{file_count} graphs ({total_flows:,} flows)...")

        missing_classes = [c for c, cnt in class_counts.items() if cnt == 0]
        benign_flows = class_counts["Benign"]
        attack_flows = total_flows - benign_flows
        info["splits"][split] = {
            "graph_count": file_count,
            "total_flows": total_flows,
            "feature_dim": feature_dim,
            "benign_flows": benign_flows,
            "attack_flows": attack_flows,
            "benign_pct": round(100.0 * benign_flows / total_flows, 4) if total_flows else 0.0,
            "attack_pct": round(100.0 * attack_flows / total_flows, 4) if total_flows else 0.0,
            "present_classes": len(cnames) - len(missing_classes),
            "missing_classes": missing_classes,
            "nan_or_inf_values": nan_or_inf_count,
            "binary_label_mismatch_graphs": binary_mismatch_graphs,
            "class_counts": class_counts,
        }

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UNSW and corrected CICIDS episode graph datasets")
    parser.add_argument("--unsw-graphs", type=Path, default=Path("graph_data_bundle/data/graphs_unsw"))
    parser.add_argument("--unsw-features", type=Path, default=Path("graph_data_bundle/data/features"))
    parser.add_argument("--cicids-graphs", type=Path, default=Path("data/graphs_cicids_episode"))
    parser.add_argument("--cicids-features", type=Path, default=Path("data/features_cicids_episode"))
    parser.add_argument("--output", type=Path, default=Path("results/physical_dataset_audit.json"))
    args = parser.parse_args()

    ref_features = None
    if (args.unsw_features / "feature_names.joblib").is_file():
        ref_features = list(joblib.load(args.unsw_features / "feature_names.joblib"))

    results: Dict[str, Any] = {}
    if args.unsw_graphs.is_dir():
        print(f"[INFO] Auditing UNSW at {args.unsw_graphs}")
        results["UNSW"] = audit_dataset("UNSW", args.unsw_graphs, args.unsw_features, ref_features)

    print(f"[INFO] Auditing CICIDS Episode at {args.cicids_graphs}")
    results["CICIDS_EPISODE"] = audit_dataset(
        "CICIDS_EPISODE", args.cicids_graphs, args.cicids_features, ref_features
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[INFO] Saved physical audit to {args.output}")


if __name__ == "__main__":
    main()
