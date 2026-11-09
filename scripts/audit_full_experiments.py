"""Comprehensive audit of saved training configurations, checkpoints, validation coverage, test coverage, class distributions, and cross-dataset transfer."""

import json
from pathlib import Path
import numpy as np

def audit_experiments():
    out = {}

    # 1. Training Runs
    train_runs = [
        ("UNSW", "Random Forest", Path("results/experiments/unsw_full_seed42/rf")),
        ("UNSW", "XGBoost", Path("results/experiments/unsw_full_seed42/xgb")),
        ("UNSW", "PyTorch MLP", Path("results/experiments/unsw_full_seed42/mlp")),
        ("CICIDS", "Random Forest", Path("results/experiments/cicids_100k_seed42/rf")),
        ("CICIDS", "XGBoost", Path("results/experiments/cicids_100k_seed42/xgb")),
        ("CICIDS", "PyTorch MLP", Path("results/experiments/cicids_100k_seed42/mlp")),
    ]

    out["training_runs"] = []
    for dname, mname, rdir in train_runs:
        cfg_file = rdir / "run_config.json"
        met_file = rdir / "metrics.json"
        ckpt_joblib = rdir / "model.joblib"
        ckpt_pt = rdir / "model.pt"

        cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        met = json.loads(met_file.read_text(encoding="utf-8")) if met_file.exists() else {}

        ckpt_path = ckpt_joblib if ckpt_joblib.exists() else (ckpt_pt if ckpt_pt.exists() else None)

        hp = cfg.get("hyperparameters", {})
        shape = cfg.get("train_matrix_shape", {})
        record = {
            "dataset": dname,
            "model": mname,
            "directory": str(rdir),
            "config_exists": cfg_file.exists(),
            "checkpoint_exists": ckpt_path is not None,
            "checkpoint_path": str(ckpt_path) if ckpt_path else None,
            "checkpoint_size_bytes": ckpt_path.stat().st_size if ckpt_path else 0,
            "seed": hp.get("seed"),
            "max_samples_arg": hp.get("max_samples"),
            "actual_train_samples": shape.get("rows"),
            "class_weight": hp.get("class_weight"),
            "train_time_seconds": met.get("train_time_seconds"),
            "sampled_train_class_distribution": cfg.get("sampled_train_class_distribution"),
            "original_train_class_distribution": cfg.get("original_train_class_distribution"),
            "hyperparameters": hp
        }
        out["training_runs"].append(record)

    # 2. Validation Coverage
    out["validation_runs"] = []
    val_runs = [
        ("UNSW", "Random Forest", Path("results/experiments/unsw_full_seed42/rf")),
        ("UNSW", "XGBoost", Path("results/experiments/unsw_full_seed42/xgb")),
        ("UNSW", "PyTorch MLP", Path("results/experiments/unsw_full_seed42/mlp")),
        ("CICIDS", "Random Forest", Path("results/experiments/cicids_100k_seed42/rf")),
        ("CICIDS", "XGBoost", Path("results/experiments/cicids_100k_seed42/xgb")),
        ("CICIDS", "PyTorch MLP", Path("results/experiments/cicids_100k_seed42/mlp")),
    ]

    for dname, mname, rdir in val_runs:
        npz_file = rdir / "predictions.npz"
        met_file = rdir / "metrics.json"

        if not npz_file.exists():
            out["validation_runs"].append({"dataset": dname, "model": mname, "error": "npz not found"})
            continue

        data = np.load(npz_file, allow_pickle=True)
        y_true_m = data["y_true_multiclass"]
        y_pred_m = data["y_pred_multiclass"]
        g_ids = data["graph_id"]
        e_ids = data["edge_id"]
        cnames = list(data["class_names"])
        benign_id = int(data["benign_class_id"])

        unique_graphs = np.unique(g_ids)
        # Check uniqueness of (graph_id, edge_id) pairs
        pair_hashes = set(zip(g_ids.tolist(), e_ids.tolist()))
        duplicates = len(g_ids) - len(pair_hashes)

        # Class counts
        cls_counts = {}
        for idx, cname in enumerate(cnames):
            cls_counts[cname] = int(np.sum(y_true_m == idx))

        out["validation_runs"].append({
            "dataset": dname,
            "model": mname,
            "evaluated_flows": len(y_true_m),
            "unique_graphs_evaluated": len(unique_graphs),
            "graph_ids_min_max": (int(np.min(g_ids)), int(np.max(g_ids))),
            "duplicate_pairs": duplicates,
            "benign_id": benign_id,
            "benign_class_name": cnames[benign_id],
            "class_distribution": cls_counts
        })

    # 3. Test Coverage
    out["test_runs"] = []
    test_runs = [
        ("UNSW", "Random Forest", Path("results/test_evaluation/unsw/rf")),
        ("UNSW", "XGBoost", Path("results/test_evaluation/unsw/xgb")),
        ("UNSW", "PyTorch MLP", Path("results/test_evaluation/unsw/mlp")),
        ("CICIDS", "Random Forest", Path("results/test_evaluation/cicids/rf")),
        ("CICIDS", "XGBoost", Path("results/test_evaluation/cicids/xgb")),
        ("CICIDS", "PyTorch MLP", Path("results/test_evaluation/cicids/mlp")),
    ]

    for dname, mname, rdir in test_runs:
        npz_file = rdir / "predictions.npz"
        met_file = rdir / "metrics.json"
        cfg_file = rdir / "run_config.json"

        if not npz_file.exists():
            out["test_runs"].append({"dataset": dname, "model": mname, "error": "npz not found"})
            continue

        data = np.load(npz_file, allow_pickle=True)
        cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        y_true_m = data["y_true_multiclass"]
        y_pred_m = data["y_pred_multiclass"]
        g_ids = data["graph_id"]
        e_ids = data["edge_id"]
        cnames = list(data["class_names"])
        benign_id = int(data["benign_class_id"])

        unique_graphs = np.unique(g_ids)
        pair_hashes = set(zip(g_ids.tolist(), e_ids.tolist()))
        duplicates = len(g_ids) - len(pair_hashes)

        cls_counts = {}
        for idx, cname in enumerate(cnames):
            cls_counts[cname] = int(np.sum(y_true_m == idx))

        out["test_runs"].append({
            "dataset": dname,
            "model": mname,
            "checkpoint_loaded": cfg.get("checkpoint_path", cfg.get("checkpoint")),
            "evaluated_flows": len(y_true_m),
            "unique_graphs_evaluated": len(unique_graphs),
            "duplicate_pairs": duplicates,
            "benign_id": benign_id,
            "class_distribution": cls_counts
        })

    # 4. Cross-Dataset Runs
    out["cross_dataset_runs"] = []
    cross_runs = [
        ("UNSW -> CICIDS", "Random Forest", Path("results/cross_dataset/unsw_to_cicids/rf")),
        ("UNSW -> CICIDS", "XGBoost", Path("results/cross_dataset/unsw_to_cicids/xgb")),
        ("UNSW -> CICIDS", "PyTorch MLP", Path("results/cross_dataset/unsw_to_cicids/mlp")),
        ("CICIDS -> UNSW", "Random Forest", Path("results/cross_dataset/cicids_to_unsw/rf")),
        ("CICIDS -> UNSW", "XGBoost", Path("results/cross_dataset/cicids_to_unsw/xgb")),
        ("CICIDS -> UNSW", "PyTorch MLP", Path("results/cross_dataset/cicids_to_unsw/mlp")),
    ]

    for direction, mname, rdir in cross_runs:
        npz_file = rdir / "predictions.npz"
        met_file = rdir / "metrics.json"
        cfg_file = rdir / "run_config.json"

        if not npz_file.exists():
            out["cross_dataset_runs"].append({"direction": direction, "model": mname, "error": "npz not found"})
            continue

        data = np.load(npz_file, allow_pickle=True)
        cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        y_true_b = data["y_true_binary"]
        y_pred_b = data["y_pred_binary"]
        g_ids = data["graph_id"]
        e_ids = data["edge_id"]

        unique_graphs = np.unique(g_ids)
        pair_hashes = set(zip(g_ids.tolist(), e_ids.tolist()))

        out["cross_dataset_runs"].append({
            "direction": direction,
            "model": mname,
            "checkpoint_loaded": cfg.get("checkpoint_path", cfg.get("checkpoint")),
            "evaluated_flows": len(y_true_b),
            "unique_graphs_evaluated": len(unique_graphs),
            "duplicate_pairs": len(y_true_b) - len(pair_hashes),
            "target_true_attacks": int(np.sum(y_true_b == 1)),
            "target_true_benign": int(np.sum(y_true_b == 0)),
        })

    out_file = Path("results/experiment_coverage_audit.json")
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[INFO] Saved experiment coverage audit to {out_file}")

if __name__ == "__main__":
    audit_experiments()
