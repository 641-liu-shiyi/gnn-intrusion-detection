"""Train an XGBoost baseline on UNSW or CICIDS graph data.

The implementation follows the same design as ``train_rf.py`` but uses the
``xgboost.XGBClassifier``.  All required artefacts for Part 2 are generated, and
the script respects the ``--output-dir`` flag to avoid overwriting historic
results.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from src.baselines.common import resolve_split_name
from src.baselines.data_loader import DataLoader
from src.eval.metrics import compute_metrics, compute_proba_metrics
from src.baselines.eval_utils import (
    write_json,
    write_csv,
    write_per_class_csv,
    write_confusion_csv,
    write_normalized_confusion_csv,
    plot_confusion_matrix,
    write_pr_curve_csv,
    plot_pr_curve,
    save_predictions_npz,
    align_probabilities,
    XGBModelWrapper,
)


def _git_commit_hash() -> str:
    """Return the short (7-char) git commit hash, or ``"unknown"`` on failure."""
    try:
        result = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return result.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on UNSW or CICIDS graph data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing 'graphs_unsw' or 'graphs_cicids'. If omitted, the default bundled path is used.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["unsw", "cicids"],
        default="unsw",
        help="Select dataset (UNSW or CICIDS).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=50, help="Number of boosting rounds")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate (eta)"
    )
    parser.add_argument("--workers", type=int, default=2, help="Parallel threads (n_jobs)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write results. If omitted, defaults to 'results/xgb_<dataset>'.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=150_000,
        help="Maximum number of training edges (0 = use all).",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="If set, evaluate the test split as well.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting strategy ('balanced' or 'none').",
    )
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else Path("results") / f"xgb_{args.dataset.lower()}"
    if out_dir.is_dir() and (out_dir / "metrics.json").is_file():
        raise RuntimeError(
            f"Output directory '{out_dir}' already contains results; abort to avoid overwriting."
        )

    split_name = resolve_split_name(args.dataset)
    loader = DataLoader(data_root=args.data_root, split_name=split_name)

    X_train, y_train_multi, train_meta = loader.load_training(max_samples=args.max_samples)
    y_train_bin = (y_train_multi != loader.benign_id).astype(np.int64)

    # Encode multiclass labels into contiguous 0..K-1 space for XGBoost
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_multi)

    # Compute per-sample class weights (balanced multiclass if requested)
    if args.class_weight == "balanced":
        classes_in_train = np.unique(y_train_enc)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes_in_train, y=y_train_enc
        )
        weight_map = dict(zip(classes_in_train, class_weights))
        sample_weight = np.vectorize(weight_map.get)(y_train_enc).astype(np.float32)
    else:
        sample_weight = None

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        tree_method="hist",
        n_jobs=args.workers,
        seed=args.seed,
    )
    start_train = time.time()
    clf.fit(X_train, y_train_enc, sample_weight=sample_weight)
    train_time = time.time() - start_train

    model = XGBModelWrapper(clf, label_encoder, len(loader.class_names))

    # Validation split with identifiers
    if hasattr(loader, "load_split_with_ids"):
        X_val, y_val_multi, y_val_bin, graph_id_val, edge_id_val = loader.load_split_with_ids("val")
    else:
        X_val, y_val_multi, y_val_bin = loader.load_split("val")
        n_val = X_val.shape[0]
        graph_id_val = np.arange(n_val, dtype=np.int64)
        edge_id_val = np.arange(n_val, dtype=np.int64)

    start_eval = time.time()
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    y_val_pred_bin = (y_val_pred != loader.benign_id).astype(np.int64)
    binary_score_val = 1.0 - y_val_proba[:, loader.benign_id]
    precision_vals, recall_vals, thr_vals = precision_recall_curve(y_val_bin, binary_score_val)
    val_metrics = compute_metrics(
        y_val_multi,
        y_val_pred,
        loader.class_names,
        binary_true=y_val_bin,
        binary_pred=y_val_pred_bin,
        benign_id=loader.benign_id,
    )
    proba_metrics = compute_proba_metrics(
        y_val_multi,
        y_val_proba,
        loader.class_names,
        benign_id=loader.benign_id,
    )
    val_metrics.update(proba_metrics)
    eval_time_val = time.time() - start_eval

    test_metrics = None
    if args.evaluate_test:
        if hasattr(loader, "load_split_with_ids"):
            X_test, y_test_multi, y_test_bin, graph_id_test, edge_id_test = loader.load_split_with_ids("test")
        else:
            X_test, y_test_multi, y_test_bin = loader.load_split("test")
            n_test = X_test.shape[0]
            graph_id_test = np.arange(n_test, dtype=np.int64)
            edge_id_test = np.arange(n_test, dtype=np.int64)
        start_test = time.time()
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        y_test_pred_bin = (y_test_pred != loader.benign_id).astype(np.int64)
        test_metrics = compute_metrics(
            y_test_multi,
            y_test_pred,
            loader.class_names,
            binary_true=y_test_bin,
            binary_pred=y_test_pred_bin,
            benign_id=loader.benign_id,
        )
        test_proba_metrics = compute_proba_metrics(
            y_test_multi,
            y_test_proba,
            loader.class_names,
            benign_id=loader.benign_id,
        )
        test_metrics.update(test_proba_metrics)
        eval_time_test = time.time() - start_test

    # Persist artefacts - reuse helper functions for consistency.
    json_path = out_dir / "metrics.json"
    json_content: Dict[str, Any] = {
        "validation": val_metrics,
        "load_time_seconds": train_meta.get("load_time_seconds", None),
        "train_time_seconds": train_time,
        "eval_time_seconds_val": eval_time_val,
    }
    if test_metrics is not None:
        json_content["test"] = test_metrics
        json_content["eval_time_seconds_test"] = eval_time_test
    write_json(json_path, json_content)

    # Flattened CSV (validation only)
    csv_path = out_dir / "metrics.csv"
    flat_rows: Dict[str, Any] = {}
    for prefix, metrics in [("val", val_metrics)]:
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                continue
            flat_rows[f"{prefix}_{key}"] = value
    flat_rows["load_time_seconds"] = train_meta.get("load_time_seconds", None)
    flat_rows["train_time_seconds"] = train_time
    flat_rows["eval_time_seconds_val"] = eval_time_val
    write_csv(csv_path, flat_rows)

    # Confusion matrix CSV
    cm_csv = out_dir / "confusion_matrix.csv"
    write_confusion_csv(cm_csv, val_metrics["confusion_matrix"], loader.class_names)

    # Per-class CSV
    write_per_class_csv(out_dir / "per_class_metrics.csv", val_metrics["per_class"])

    # Normalised confusion CSV & PNG
    norm_cm_csv = out_dir / "confusion_matrix_normalized.csv"
    write_normalized_confusion_csv(norm_cm_csv, val_metrics["confusion_matrix"], loader.class_names)
    plot_confusion_matrix(
        out_dir / "confusion_matrix_normalized.png",
        val_metrics["confusion_matrix"],
        loader.class_names,
        title="Normalized Confusion matrix",
    )
    # Raw confusion PNG
    plot_confusion_matrix(
        out_dir / "confusion_matrix.png",
        val_metrics["confusion_matrix"],
        loader.class_names,
        title="Confusion matrix",
    )

    # Binary confusion matrix CSV and PNG
    binary_cm = confusion_matrix(y_val_bin, y_val_pred_bin)
    write_confusion_csv(
        out_dir / "confusion_matrix_binary.csv",
        binary_cm.tolist(),
        ["Benign", "Attack"],
    )
    plot_confusion_matrix(
        out_dir / "confusion_matrix_binary.png",
        binary_cm.tolist(),
        ["Benign", "Attack"],
        title="Binary confusion matrix",
    )

    # PR curve CSV + PNG (binary)
    write_pr_curve_csv(out_dir / "precision_recall_curve.csv", precision_vals, recall_vals, thr_vals)
    plot_pr_curve(out_dir / "precision_recall_curve.png", precision_vals, recall_vals, title="Precision-Recall curve")

    # Prediction export (validation)
    save_predictions_npz(
        out_dir / "predictions.npz",
        y_true_multiclass=y_val_multi,
        y_pred_multiclass=y_val_pred,
        y_true_binary=y_val_bin,
        y_pred_binary=y_val_pred_bin,
        y_proba_multiclass=y_val_proba,
        graph_id=graph_id_val,
        edge_id=edge_id_val,
        class_names=loader.class_names,
        benign_class_id=loader.benign_id,
        dataset=args.dataset,
        split="val",
    )

    # ---------------------------------------------------------------------
    # Save the trained XGBoost model checkpoint for cross-dataset evaluation
    # ---------------------------------------------------------------------
    checkpoint_path = out_dir / "model.joblib"
    try:
        import joblib

        joblib.dump(model, checkpoint_path)
    except Exception as exc:  # pragma: no cover - checkpoint saving should succeed under normal conditions
        print(f"[WARN] Failed to save XGB checkpoint to {checkpoint_path}: {exc}")

    # Run configuration metadata
    run_cfg_path = out_dir / "run_config.json"
    try:
        test_graphs = len(loader._list_graph_files("test"))
    except FileNotFoundError:
        test_graphs = 0
    split_counts = {
        "train_graphs": len(loader._list_graph_files("train")),
        "val_graphs": len(loader._list_graph_files("val")),
        "test_graphs": test_graphs,
    }
    total_flows = {
        "train": X_train.shape[0],
        "val": X_val.shape[0],
        "test": X_test.shape[0] if args.evaluate_test else None,
    }
    run_config: Dict[str, Any] = {
        "dataset": args.dataset.upper(),
        "feature_count": 39,
        "class_names": loader.class_names,
        "benign_id": loader.benign_id,
        "split_graph_counts": split_counts,
        "total_flow_counts": total_flows,
        "original_train_class_distribution": train_meta.get("original_class_counts", {}),
        "sampled_train_class_distribution": train_meta.get("sampled_class_counts", {}),
        "sampling_policy": train_meta.get("sampling_policy", ""),
        "train_matrix_shape": {"rows": X_train.shape[0], "cols": X_train.shape[1]},
        "model_n_features_in": getattr(clf, "n_features_in_", None),
        "hyperparameters": {
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "workers": args.workers,
            "max_samples": args.max_samples,
            "class_weight": args.class_weight,
            "sample_weighting": args.class_weight,
        },
        "git_commit": _git_commit_hash(),
        "python_version": platform.python_version(),
        "xgboost_version": xgb.__version__,
        "timings_seconds": {
            "train": train_time,
            "eval_val": eval_time_val,
            "eval_test": eval_time_test if args.evaluate_test else None,
        },
    }
    write_json(run_cfg_path, run_config)

    print("[INFO] All results written to", out_dir)


if __name__ == "__main__":
    main()