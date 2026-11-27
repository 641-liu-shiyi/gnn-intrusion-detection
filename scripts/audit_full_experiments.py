"""Comprehensive audit & independent prediction recomputation for Part 2 experiments (UNSW + CICIDS Episode)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from src.eval.metrics import compute_metrics, compute_proba_metrics


def _recompute_run(npz_path: Path, expected_dict: Dict[str, Any], is_binary_cross: bool = False) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        y_true_m = data["y_true_multiclass"]
        y_pred_m = data["y_pred_multiclass"]
        y_true_b = data["y_true_binary"]
        y_pred_b = data["y_pred_binary"]
        y_proba = data["y_proba_multiclass"]
        g_ids = data["graph_id"]
        e_ids = data["edge_id"]
        cnames = [str(x) for x in data["class_names"]]
        benign_id = int(data["benign_class_id"])

    n = len(y_true_b)
    stride = int(np.max(e_ids)) + 1 if n > 0 else 1
    unique_pairs = int(np.unique(g_ids.astype(np.int64) * stride + e_ids.astype(np.int64)).size)
    duplicate_pairs = n - unique_pairs

    if is_binary_cross:
        recomp = compute_metrics(
            y_true_b,
            y_pred_b,
            class_names=["Benign", "Attack"],
            binary_true=y_true_b,
            binary_pred=y_pred_b,
            benign_id=0,
        )
        recomp.update(
            compute_proba_metrics(
                y_true_b,
                y_proba,
                class_names=["Benign", "Attack"],
                benign_id=0,
                binary_true=y_true_b,
            )
        )
        cm_recomp = confusion_matrix(y_true_b, y_pred_b).tolist()
    else:
        recomp = compute_metrics(
            y_true_m,
            y_pred_m,
            class_names=cnames,
            binary_true=y_true_b,
            binary_pred=y_pred_b,
            benign_id=benign_id,
        )
        recomp.update(
            compute_proba_metrics(
                y_true_m,
                y_proba,
                class_names=cnames,
                benign_id=benign_id,
                binary_true=y_true_b,
            )
        )
        cm_recomp = recomp["confusion_matrix"]

    scalar_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "binary_precision",
        "binary_recall",
        "binary_f1",
        "binary_average_precision",
    ]
    max_diff = 0.0
    for k in scalar_keys:
        if k in expected_dict and k in recomp:
            max_diff = max(max_diff, abs(float(expected_dict[k]) - float(recomp[k])))

    cm_match = True
    if not is_binary_cross and "confusion_matrix" in expected_dict:
        cm_match = expected_dict["confusion_matrix"] == cm_recomp

    return {
        "evaluated_flows": n,
        "unique_graphs_evaluated": int(len(np.unique(g_ids))),
        "duplicate_pairs": duplicate_pairs,
        "benign_id": benign_id,
        "max_scalar_discrepancy": max_diff,
        "confusion_matrix_exact_match": cm_match,
    }


def audit_experiments() -> None:
    out: Dict[str, Any] = {"training_runs": [], "validation_runs": [], "test_runs": [], "cross_dataset_runs": []}

    train_runs: List[Tuple[str, str, Path]] = [
        ("UNSW", "Random Forest", Path("results/experiments/unsw_full_seed42/rf")),
        ("UNSW", "XGBoost", Path("results/experiments/unsw_full_seed42/xgb")),
        ("UNSW", "PyTorch MLP", Path("results/experiments/unsw_full_seed42/mlp")),
        ("CICIDS_EPISODE", "Random Forest", Path("results/experiments/cicids_episode_100k_seed42/rf")),
        ("CICIDS_EPISODE", "XGBoost", Path("results/experiments/cicids_episode_100k_seed42/xgb")),
        ("CICIDS_EPISODE", "PyTorch MLP", Path("results/experiments/cicids_episode_100k_seed42/mlp")),
    ]

    for dname, mname, rdir in train_runs:
        cfg_file = rdir / "run_config.json"
        met_file = rdir / "metrics.json"
        ckpt_path = rdir / "model.joblib" if (rdir / "model.joblib").exists() else rdir / "model.pt"
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        met = json.loads(met_file.read_text(encoding="utf-8"))
        hp = cfg.get("hyperparameters", {})
        shape = cfg.get("train_matrix_shape", {})
        out["training_runs"].append(
            {
                "dataset": dname,
                "model": mname,
                "directory": str(rdir),
                "split_name": cfg.get("split_name"),
                "checkpoint_exists": ckpt_path.exists(),
                "seed": hp.get("seed"),
                "actual_train_samples": shape.get("rows"),
                "class_weight": hp.get("class_weight"),
                "train_time_seconds": met.get("train_time_seconds"),
            }
        )

        val_audit = _recompute_run(rdir / "predictions.npz", met.get("validation", met), is_binary_cross=False)
        out["validation_runs"].append({"dataset": dname, "model": mname, **val_audit})

    test_runs: List[Tuple[str, str, Path]] = [
        ("UNSW", "Random Forest", Path("results/test_evaluation/unsw/rf")),
        ("UNSW", "XGBoost", Path("results/test_evaluation/unsw/xgb")),
        ("UNSW", "PyTorch MLP", Path("results/test_evaluation/unsw/mlp")),
        ("CICIDS_EPISODE", "Random Forest", Path("results/test_evaluation/cicids_episode/rf")),
        ("CICIDS_EPISODE", "XGBoost", Path("results/test_evaluation/cicids_episode/xgb")),
        ("CICIDS_EPISODE", "PyTorch MLP", Path("results/test_evaluation/cicids_episode/mlp")),
    ]
    for dname, mname, rdir in test_runs:
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        test_audit = _recompute_run(rdir / "predictions.npz", met.get("evaluation", met), is_binary_cross=False)
        out["test_runs"].append({"dataset": dname, "model": mname, **test_audit})

    cross_runs: List[Tuple[str, str, str, Path]] = [
        ("UNSW -> CICIDS (Episode)", "val", m, Path(f"results/cross_dataset_episode/unsw_to_cicids_val/{m}"))
        for m in ("rf", "xgb", "mlp")
    ] + [
        ("UNSW -> CICIDS (Episode)", "test", m, Path(f"results/cross_dataset_episode/unsw_to_cicids_test/{m}"))
        for m in ("rf", "xgb", "mlp")
    ] + [
        ("CICIDS (Episode) -> UNSW", "val", m, Path(f"results/cross_dataset_episode/cicids_to_unsw_val/{m}"))
        for m in ("rf", "xgb", "mlp")
    ] + [
        ("CICIDS (Episode) -> UNSW", "test", m, Path(f"results/cross_dataset_episode/cicids_to_unsw_test/{m}"))
        for m in ("rf", "xgb", "mlp")
    ]
    for direction, split, mname, rdir in cross_runs:
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        cross_audit = _recompute_run(rdir / "predictions.npz", met, is_binary_cross=True)
        out["cross_dataset_runs"].append({"direction": direction, "split": split, "model": mname, **cross_audit})

    out_file = Path("results/experiment_coverage_audit.json")
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[INFO] Saved experiment coverage & recomputation audit to {out_file}")

    all_audits = out["validation_runs"] + out["test_runs"] + out["cross_dataset_runs"]
    max_err = max(a["max_scalar_discrepancy"] for a in all_audits)
    dup_total = sum(a["duplicate_pairs"] for a in all_audits)
    cm_ok = all(a["confusion_matrix_exact_match"] for a in all_audits)
    print(f"[AUDIT] Max metric recomputation discrepancy: {max_err:.8f}, total duplicates: {dup_total}, confusion matrices exact: {cm_ok}")
    if max_err > 1e-9 or dup_total > 0 or not cm_ok:
        raise RuntimeError("Audit verification failed!")


if __name__ == "__main__":
    audit_experiments()
