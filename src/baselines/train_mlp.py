"""Train a simple PyTorch MLP baseline on the UNSW graph dataset.

The implementation mirrors the Random-Forest and XGBoost scripts but uses a
feed-forward neural network defined in ``src.baselines.mlp_model``.  The training
pipeline follows the project specifications:

* 39 input features (the ``edge_attr`` columns)
* Hidden layers: 64 -> 32 with ReLU activations and ``dropout=0.2``
* Output layer: 10 classes (multiclass)
* Weighted ``CrossEntropyLoss`` where class weights follow the scikit-learn
  ``balanced`` strategy.
* Adam optimiser, learning-rate 0.001
* Mini-batch size 256, maximum 10 epochs, early stopping patience 3 (based on
  validation loss)
* Validation (and optional test) evaluation using the shared ``compute_metrics``
  utility.
* All results are written under ``results/mlp_unsw/`` in the same format as the
  other baseline scripts.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# ---------------------------------------------------------------------------
# External library imports
# ---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Project-internal imports
# ---------------------------------------------------------------------------
from src.eval.metrics import compute_metrics, compute_proba_metrics
from src.baselines.data_loader import DataLoader
from src.baselines.mlp_model import MLP
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
)


# ---------------------------------------------------------------------------
def _git_commit_hash() -> str:
    """Return the short (7-char) git commit hash, or ``"unknown"`` on failure."""
    try:
        result = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return result.strip()
    except Exception:
        return "unknown"


# The original script defined private helper functions for JSON/CSV writing.
# They have been replaced by the shared utilities in ``src.baselines.eval_utils``.
# The definitions are therefore removed to avoid duplication and potential
# inconsistencies.


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP on UNSW or CICIDS graph data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing the graph datasets. If omitted, the default bundled path is used.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["unsw", "cicids"],
        default="unsw",
        help="Select dataset (UNSW or CICIDS). Determines which graph directory is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=150_000,
        help="Maximum number of training edges (0 = use all).",
    )
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience (epochs)")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--workers", type=int, default=2, help="Num torch DataLoader workers")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability after each hidden layer",
    )
    parser.add_argument("--hidden1", type=int, default=64, help="Size of first hidden layer")
    parser.add_argument("--hidden2", type=int, default=32, help="Size of second hidden layer")
    parser.add_argument(
        "--class_weight",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting strategy ('balanced' or 'none')",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="If set, evaluate the final model on the test split as well.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write results. If omitted, defaults to 'results/mlp_<dataset>'.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Seed everything for reproducibility
    # ---------------------------------------------------------------
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    # Resolve the sub-directory name for the chosen dataset (UNSW or CICIDS).
    # ``resolve_split_name`` is defined in ``src.baselines.common`` and mirrors the
    # behaviour of the RF/XGB scripts.
    from src.baselines.common import resolve_split_name

    # Resolve the correct sub-directory based on the chosen dataset (UNSW or CICIDS).
    # ``resolve_split_name`` returns e.g. "graphs_unsw" which the ``DataLoader`` expects.
    split_name = resolve_split_name(args.dataset)
    loader = DataLoader(data_root=args.data_root, split_name=split_name)
    X_train_np, y_train_np, train_meta = loader.load_training(max_samples=args.max_samples)
    X_val_np, y_val_multi_np, y_val_binary_np = loader.load_split("val")
    # Load validation IDs for export - some mock loaders used in tests may not
    # implement ``load_split_with_ids``.  Fall back to sequential IDs when needed.
    if hasattr(loader, "load_split_with_ids"):
        _, _, _, graph_id_val, edge_id_val = loader.load_split_with_ids("val")
    else:
        n_val = X_val_np.shape[0]
        graph_id_val = np.arange(n_val, dtype=np.int64)
        edge_id_val = np.arange(n_val, dtype=np.int64)
    if args.evaluate_test:
        X_test_np, y_test_multi_np, y_test_binary_np = loader.load_split("test")
    else:
        X_test_np, y_test_multi_np, y_test_binary_np = None, None, None

    # ---------------------------------------------------------------
    # PyTorch tensors and DataLoader
    # ---------------------------------------------------------------
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()

    # Compute per-class weights (balanced) matching the full output class space
    num_classes = len(loader.class_names)
    if args.class_weight == "balanced":
        present_classes = np.unique(y_train_np)
        raw_weights = compute_class_weight(
            class_weight="balanced", classes=present_classes, y=y_train_np
        )
        full_weights = np.ones(num_classes, dtype=np.float32)
        for cls_id, w in zip(present_classes, raw_weights):
            if 0 <= cls_id < num_classes:
                full_weights[int(cls_id)] = float(w)
        class_weight_tensor = torch.from_numpy(full_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    # Validation tensors (no batching needed - the set is small enough)
    X_val = torch.from_numpy(X_val_np).float()
    y_val = torch.from_numpy(y_val_multi_np).long()

    if args.evaluate_test:
        X_test = torch.from_numpy(X_test_np).float()
        y_test = torch.from_numpy(y_test_multi_np).long()

    # ---------------------------------------------------------------
    # Model, loss, optimiser
    # ---------------------------------------------------------------
    model = MLP(
        input_dim=X_train_np.shape[1],
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        output_dim=num_classes,
        dropout=args.dropout,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ---------------------------------------------------------------
    # Training loop with early stopping based on validation loss
    # ---------------------------------------------------------------
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improve = 0

    start_train = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * xb.size(0)
        epoch_loss = epoch_loss_sum / X_train.shape[0]

        # -----------------------------------------------------------
        # Validation evaluation
        # -----------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
        # Early-stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= args.patience:
            # No improvement for ``patience`` epochs - stop training
            break
    train_time = time.time() - start_train

    # ---------------------------------------------------------------
    # Restore best model
    # ---------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------------------------------------------------------
    # Final validation metrics (and optional test metrics)
    # ---------------------------------------------------------------
    start_eval = time.time()
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_proba = torch.nn.functional.softmax(val_logits, dim=1).cpu().numpy()
        val_pred_multi = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_pred_binary = (val_pred_multi != loader.benign_id).astype(np.int64)
        val_metrics = compute_metrics(
            y_true=y_val_multi_np,
            y_pred=val_pred_multi,
            class_names=loader.class_names,
            benign_id=loader.benign_id,
        )
        val_proba_metrics = compute_proba_metrics(
            y_val_multi_np,
            val_proba,
            loader.class_names,
            benign_id=loader.benign_id,
        )
        val_metrics.update(val_proba_metrics)

        if args.evaluate_test:
            test_logits = model(X_test)
            test_proba = torch.nn.functional.softmax(test_logits, dim=1).cpu().numpy()
            test_pred_multi = torch.argmax(test_logits, dim=1).cpu().numpy()
            test_metrics = compute_metrics(
                y_true=y_test_multi_np,
                y_pred=test_pred_multi,
                class_names=loader.class_names,
                benign_id=loader.benign_id,
            )
            test_proba_metrics = compute_proba_metrics(
                y_test_multi_np,
                test_proba,
                loader.class_names,
                benign_id=loader.benign_id,
            )
            test_metrics.update(test_proba_metrics)
        else:
            test_metrics = None
    eval_time = time.time() - start_eval

    # ---------------------------------------------------------------
    # Persist results
    # ---------------------------------------------------------------
    # Resolve output directory - honour user-supplied path, otherwise use a default per-dataset.
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = Path("results") / f"mlp_{args.dataset.lower()}"
    # Guard against accidental overwrites: if the directory exists and already contains a metrics JSON we consider it a completed run.
    if out_dir.is_dir() and (out_dir / "metrics.json").is_file():
        raise RuntimeError(f"Output directory '{out_dir}' already contains results; abort to avoid overwriting.")
    # 1. JSON metrics (validation + optional test)
    metrics_json_path = out_dir / "metrics.json"
    json_content: Dict[str, Any] = {
        "validation": val_metrics,
        "load_time_seconds": train_meta.get("load_time_seconds", None),
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
    }
    if test_metrics is not None:
        json_content["test"] = test_metrics
    write_json(metrics_json_path, json_content)

    # 2. Flattened CSV (validation only for brevity)
    metrics_csv_path = out_dir / "metrics.csv"
    flat_rows: Dict[str, Any] = {}
    for prefix, metrics in [("val", val_metrics)]:
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                continue
            flat_rows[f"{prefix}_{key}"] = value
    flat_rows["load_time_seconds"] = train_meta.get("load_time_seconds", None)
    flat_rows["train_time_seconds"] = train_time
    flat_rows["eval_time_seconds"] = eval_time
    write_csv(metrics_csv_path, flat_rows)

    # ---------------------------------------------------------------------
    # Additional artefacts required for Part 2 (validation split only)
    # ---------------------------------------------------------------------
    # Per-class metrics CSV
    write_per_class_csv(out_dir / "per_class_metrics.csv", val_metrics["per_class"])

    # Normalised confusion CSV and PNG
    norm_cm_csv = out_dir / "confusion_matrix_normalized.csv"
    write_normalized_confusion_csv(norm_cm_csv, val_metrics["confusion_matrix"], loader.class_names)
    plot_confusion_matrix(
        out_dir / "confusion_matrix_normalized.png",
        val_metrics["confusion_matrix"],
        loader.class_names,
        title="Normalized Confusion matrix",
    )

    # Raw confusion matrix PNG (validation)
    plot_confusion_matrix(
        out_dir / "confusion_matrix.png",
        val_metrics["confusion_matrix"],
        loader.class_names,
        title="Confusion matrix",
    )

    # Binary confusion matrix CSV and PNG (derived from validation predictions)
    binary_cm = confusion_matrix(y_val_binary_np, val_pred_binary)
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

    # Precision-Recall curve (binary) CSV + PNG
    # Compute binary scores from softmax probabilities (attack probability = 1 - P(Benign))
    val_proba = torch.nn.functional.softmax(val_logits, dim=1).cpu().numpy()
    prob_benign = val_proba[:, loader.benign_id]
    binary_score = 1.0 - prob_benign
    precision_vals, recall_vals, thr_vals = precision_recall_curve(y_val_binary_np, binary_score)
    write_pr_curve_csv(
        out_dir / "precision_recall_curve.csv",
        precision_vals,
        recall_vals,
        thr_vals,
    )
    plot_pr_curve(
        out_dir / "precision_recall_curve.png",
        precision_vals,
        recall_vals,
        title="Precision-Recall curve",
    )
    # Prediction export (validation)
    save_predictions_npz(
        out_dir / "predictions.npz",
        y_true_multiclass=y_val_multi_np,
        y_pred_multiclass=val_pred_multi,
        y_true_binary=y_val_binary_np,
        y_pred_binary=val_pred_binary,
        y_proba_multiclass=val_proba,
        graph_id=graph_id_val,
        edge_id=edge_id_val,
        class_names=loader.class_names,
        benign_class_id=loader.benign_id,
        dataset=args.dataset,
        split="val",
    )

    # ---------------------------------------------------------------------
    # Save the trained MLP model checkpoint for cross-dataset evaluation
    # ---------------------------------------------------------------------
    checkpoint_path = out_dir / "model.pt"
    try:
        # torch.save the state dict; the model architecture can be reconstructed
        # from the source feature dimension and class count during loading.
        torch.save(model.state_dict(), checkpoint_path)
    except Exception as exc:  # pragma: no cover - should not happen under normal conditions
        print(f"[WARN] Failed to save MLP checkpoint to {checkpoint_path}: {exc}")

    # 3. Confusion matrix CSV (test only when evaluated)
    if test_metrics is not None:
        cm_path = out_dir / "confusion_matrix.csv"
        write_confusion_csv(cm_path, test_metrics["confusion_matrix"], loader.class_names)

    # 4. Run configuration JSON
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
        "original_train_class_distribution": train_meta["original_class_counts"],
        "sampled_train_class_distribution": train_meta["sampled_class_counts"],
        "sampling_policy": train_meta["sampling_policy"],
        "train_matrix_shape": {"rows": X_train.shape[0], "cols": X_train.shape[1]},
        "model_n_features_in": model.net[0].in_features,  # input dimension = 39
        "hyperparameters": {
            "seed": args.seed,
            "max_samples": args.max_samples,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "workers": args.workers,
            "dropout": args.dropout,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "class_weight": args.class_weight,
        },
        "git_commit": _git_commit_hash(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "timings_seconds": {
            "load": train_meta.get("load_time_seconds", None),
            "train": train_time,
            "evaluation": eval_time,
        },
    }
    write_json(run_cfg_path, run_config)

    print("[INFO] All results written to", out_dir)


if __name__ == "__main__":
    main()
