"""Train a RandomForest baseline on UNSW or CICIDS graph data.

This script mirrors the behaviour of the original implementation but adds:

* a proper ``main()`` entry point with argument parsing.
* optional ``--output-dir`` to avoid overwriting historic pilot results.
* generation of the full suite of evaluation artefacts required for Part 2:
  ``metrics.json``, ``metrics.csv``, per-class CSV, raw and normalised confusion
  matrices (CSV + PNG), precision-recall curve (CSV + PNG) and a ``predictions.npz``
  export containing true/predicted labels, probabilities and graph/edge IDs.

The implementation deliberately re-uses the helper utilities in
``src.baselines.eval_utils`` to keep the code concise and consistent across the
three baseline scripts.
"""

from __future__ import annotations

import argparse
import json
import time
import subprocess
import platform
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.baselines.data_loader import DataLoader
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
)
from sklearn.metrics import confusion_matrix, precision_recall_curve
from src.eval.metrics import compute_metrics, compute_proba_metrics
from typing import Any, Dict
from src.baselines.common import resolve_split_name


def _git_commit_hash() -> str:
    """Return the short (7-char) git commit hash, or ``"unknown"`` on failure."""
    try:
        result = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return result.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RandomForest on UNSW or CICIDS graph data")
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
        help="Select dataset (UNSW or CICIDS). Determines which graph directory is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=50, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum depth of each tree")
    parser.add_argument(
        "--min_samples_leaf", type=int, default=2, help="Minimum samples required at a leaf node"
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel jobs (n_jobs)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write results. If omitted, defaults to 'results/rf_<dataset>'.",
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
        help="If set, evaluate the test split as well (additional artefacts will be written).",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        choices=["balanced_subsample", "balanced", "none"],
        default="balanced_subsample",
        help="Class weighting strategy ('balanced_subsample', 'balanced', or 'none').",
    )
    args = parser.parse_args()

    # Resolve the output directory and guard against accidental overwrites.
    out_dir = args.output_dir if args.output_dir is not None else Path("results") / f"rf_{args.dataset.lower()}"
    if out_dir.is_dir() and (out_dir / "metrics.json").is_file():
        raise RuntimeError(
            f"Output directory '{out_dir}' already contains results; abort to avoid overwriting."
        )

    # Resolve the appropriate sub-directory for the selected dataset.
    split_name = resolve_split_name(args.dataset)
    loader = DataLoader(data_root=args.data_root, split_name=split_name)

    # ---------------------------------------------------------------------
    # Load training data (may be subsampled via ``--max_samples``)
    # ---------------------------------------------------------------------
    X_train, y_train_multi, train_meta = loader.load_training(max_samples=args.max_samples)
    y_train_bin = (y_train_multi != loader.benign_id).astype(np.int64)

    # ---------------------------------------------------------------------
    # Train the RandomForest model
    # ---------------------------------------------------------------------
    rf_class_weight = None if args.class_weight == "none" else args.class_weight
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.workers,
        class_weight=rf_class_weight,
        random_state=args.seed,
    )
    start_train = time.time()
    rf.fit(X_train, y_train_multi)
    train_time = time.time() - start_train

    # ---------------------------------------------------------------------
    # Validation split - with identifiers for export
    # ---------------------------------------------------------------------
    if hasattr(loader, "load_split_with_ids"):
        X_val, y_val_multi, y_val_bin, graph_id_val, edge_id_val = loader.load_split_with_ids("val")
    else:
        X_val, y_val_multi, y_val_bin = loader.load_split("val")
        n_val = X_val.shape[0]
        graph_id_val = np.arange(n_val, dtype=np.int64)
        edge_id_val = np.arange(n_val, dtype=np.int64)

    start_eval = time.time()
    y_val_pred = rf.predict(X_val)
    y_val_proba = align_probabilities(rf.predict_proba(X_val), rf.classes_, len(loader.class_names))
    y_val_pred_bin = (y_val_pred != loader.benign_id).astype(np.int64)
    binary_score_val = 1.0 - y_val_proba[:, loader.benign_id]
    precision_vals, recall_vals, thr_vals = precision_recall_curve(y_val_bin, binary_score_val)

    # Multiclass metrics (including per-class dict and confusion matrix)
    val_metrics = compute_metrics(
        y_val_multi,
        y_val_pred,
        loader.class_names,
        binary_true=y_val_bin,
        binary_pred=y_val_pred_bin,
        benign_id=loader.benign_id,
    )
    # Probability-based metrics (binary AP and per-class AP)
    proba_metrics = compute_proba_metrics(
        y_val_multi,
        y_val_proba,
        loader.class_names,
        benign_id=loader.benign_id,
    )
    val_metrics.update(proba_metrics)
    eval_time_val = time.time() - start_eval

    # ---------------------------------------------------------------------
    # Optional test split
    # ---------------------------------------------------------------------
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
        y_test_pred = rf.predict(X_test)
        y_test_pred_bin = (y_test_pred != loader.benign_id).astype(np.int64)
        test_metrics = compute_metrics(
            y_test_multi,
            y_test_pred,
            loader.class_names,
            binary_true=y_test_bin,
            binary_pred=y_test_pred_bin,
            benign_id=loader.benign_id,
        )
        eval_time_test = time.time() - start_test

    # ---------------------------------------------------------------------
    # Persist all artefacts under ``out_dir``
    # ---------------------------------------------------------------------
    # 1. JSON with both splits (test may be omitted)
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

    # 2. Flattened CSV (validation only for brevity)
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

    # 3. Confusion matrix CSV (validation)
    cm_csv_path = out_dir / "confusion_matrix.csv"
    write_confusion_csv(cm_csv_path, val_metrics["confusion_matrix"], loader.class_names)

    # 4. Per-class CSV
    per_class_path = out_dir / "per_class_metrics.csv"
    write_per_class_csv(per_class_path, val_metrics["per_class"])

    # 5. Normalised confusion CSV and PNG
    norm_cm_csv = out_dir / "confusion_matrix_normalized.csv"
    write_normalized_confusion_csv(norm_cm_csv, val_metrics["confusion_matrix"], loader.class_names)
    norm_cm_png = out_dir / "confusion_matrix_normalized.png"
    plot_confusion_matrix(norm_cm_png, val_metrics["confusion_matrix"], loader.class_names, title="Normalized Confusion matrix")

    # 6. Raw confusion matrix PNG
    cm_png = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm_png, val_metrics["confusion_matrix"], loader.class_names, title="Confusion matrix")

    # 6b. Binary confusion matrix CSV and PNG
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

    # 7. PR curve CSV and PNG (binary)
    pr_csv = out_dir / "precision_recall_curve.csv"
    write_pr_curve_csv(pr_csv, precision_vals, recall_vals, thr_vals)
    pr_png = out_dir / "precision_recall_curve.png"
    plot_pr_curve(pr_png, precision_vals, recall_vals, title="Precision-Recall curve")

    # 8. Prediction export (validation only)
    pred_npz_path = out_dir / "predictions.npz"
    save_predictions_npz(
        pred_npz_path,
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
    # Save the trained RandomForest model checkpoint for cross-dataset evaluation
    # ---------------------------------------------------------------------
    checkpoint_path = out_dir / "model.joblib"
    try:
        import joblib

        joblib.dump(rf, checkpoint_path)
    except Exception as exc:  # pragma: no cover - checkpoint saving should succeed under normal conditions
        print(f"[WARN] Failed to save RF checkpoint to {checkpoint_path}: {exc}")

    # 9. Run configuration JSON (metadata for reproducibility)
    run_cfg_path = out_dir / "run_config.json"
    # ``_list_graph_files`` raises ``FileNotFoundError`` if a split has no files.
    # The training script should still succeed when the optional ``test`` split
    # is missing (as in the unit tests).  Guard against the exception and treat
    # a missing test split as having zero graphs.
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
        "model_n_features_in": rf.n_features_in_,
        "hyperparameters": {
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "workers": args.workers,
            "max_samples": args.max_samples,
            "class_weight": args.class_weight,
        },
        "git_commit": _git_commit_hash(),
        "python_version": platform.python_version(),
        "sklearn_version": "".join(["%s." % v for v in __import__("sklearn").__version__.split('.')][:3]),
        "timings_seconds": {
            "load": train_meta.get("load_time_seconds", None),
            "train": train_time,
            "eval_val": eval_time_val,
            "eval_test": eval_time_test if args.evaluate_test else None,
        },
    }
    write_json(run_cfg_path, run_config)

    print("[INFO] All results written to", out_dir)


if __name__ == "__main__":
    main()