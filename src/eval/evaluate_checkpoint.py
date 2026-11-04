"""Standalone evaluation of a saved baseline model checkpoint on validation or held-out test split.

This module performs evaluation ONLY - it never trains or refits any model or scaler.
It guarantees that:
* Checkpoints are reloaded directly from disk without modification.
* Hyperparameters and classification thresholds are frozen.
* When evaluating on the held-out test split, `--confirm-test` is explicitly required.
* Authentic graph and edge identifiers are preserved in `predictions.npz`.
* A complete artifact suite is generated (metrics.json, metrics.csv, run_config.json,
  predictions.npz, per_class_metrics.csv, multiclass & binary confusion matrices,
  precision-recall curves).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.baselines.common import resolve_split_name
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
    align_probabilities,
    XGBModelWrapper,
)
from src.eval.metrics import compute_metrics, compute_proba_metrics

# Ensure XGBModelWrapper is available for unpickling
import __main__
setattr(__main__, "XGBModelWrapper", XGBModelWrapper)


def _load_model(model_type: str, checkpoint_path: Path, num_classes: int) -> Any:
    """Load a frozen model checkpoint from disk."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if model_type in {"rf", "xgb"}:
        import joblib
        model = joblib.load(checkpoint_path)
        if not hasattr(model, "predict"):
            raise ValueError(f"Loaded {model_type} checkpoint does not have a predict method")
        return model

    if model_type == "mlp":
        state = torch.load(checkpoint_path, map_location="cpu")
        hidden1 = state["net.0.weight"].shape[0] if "net.0.weight" in state else 64
        hidden2 = state["net.3.weight"].shape[0] if "net.3.weight" in state else 32
        out_dim = state["net.6.weight"].shape[0] if "net.6.weight" in state else num_classes
        model = MLP(input_dim=39, hidden1=hidden1, hidden2=hidden2, output_dim=out_dim)
        model.load_state_dict(state)
        model.eval()
        return model

    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen model checkpoint on validation or test data")
    parser.add_argument("--model", choices=["rf", "xgb", "mlp"], required=True, help="Model type")
    parser.add_argument("--dataset", choices=["unsw", "cicids"], required=True, help="Dataset (UNSW or CICIDS)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Split to evaluate (default: test)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for evaluation artifacts")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing graph data")
    parser.add_argument("--confirm-test", action="store_true", help="Explicit confirmation required to evaluate test split")

    args = parser.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise ValueError("Evaluating on the held-out test split requires the explicit --confirm-test flag")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists():
        raise RuntimeError(f"Output directory {out_dir} already contains results; aborting to prevent overwrite")

    # Resolve split and data loader
    split_name = resolve_split_name(args.dataset)
    loader = DataLoader(data_root=args.data_root, split_name=split_name)
    num_classes = len(loader.class_names)

    print(f"[INFO] Loading split '{args.split}' for dataset '{args.dataset.upper()}'...")
    X, y_multi, y_bin, graph_ids, edge_ids = loader.load_split_with_ids(args.split)
    n_samples = X.shape[0]
    print(f"[INFO] Loaded {n_samples} flows across {len(np.unique(graph_ids))} graphs.")

    # Load frozen model checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    model = _load_model(args.model, args.checkpoint, num_classes)

    # Perform inference and record timings
    print(f"[INFO] Running inference on {n_samples} flows...")
    start_eval = time.time()

    if args.model in {"rf", "xgb"}:
        y_pred = model.predict(X)
        raw_proba = model.predict_proba(X)
        model_classes = getattr(model, "classes_", np.arange(num_classes))
        y_proba = align_probabilities(raw_proba, model_classes, num_classes)
    else:  # mlp
        with torch.no_grad():
            tensor_X = torch.from_numpy(X).float()
            logits = model(tensor_X)
            y_proba = F.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(y_proba, axis=1)

    inference_duration = max(time.time() - start_eval, 1e-6)
    print(f"[INFO] Inference completed in {inference_duration:.2f} seconds ({n_samples / inference_duration:.1f} flows/sec).")

    # Binary conversion
    y_pred_bin = (y_pred != loader.benign_id).astype(np.int64)
    prob_benign = y_proba[:, loader.benign_id]
    binary_score = 1.0 - prob_benign
    precision_vals, recall_vals, thr_vals = precision_recall_curve(y_bin, binary_score)

    # Compute metrics
    metrics = compute_metrics(
        y_multi,
        y_pred,
        loader.class_names,
        binary_true=y_bin,
        binary_pred=y_pred_bin,
        benign_id=loader.benign_id,
    )
    proba_metrics = compute_proba_metrics(
        y_multi,
        y_proba,
        loader.class_names,
        benign_id=loader.benign_id,
        binary_true=y_bin,
    )
    metrics.update(proba_metrics)

    evaluation_report = {
        "evaluation": metrics,
        "n_samples": n_samples,
        "dataset": args.dataset.upper(),
        "split": args.split,
        "model_type": args.model,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "inference_duration_seconds": inference_duration,
        "flows_per_second": n_samples / max(inference_duration, 1e-6),
    }

    # Write artifacts
    write_json(out_dir / "metrics.json", evaluation_report)
    write_csv(out_dir / "metrics.csv", metrics)
    write_per_class_csv(out_dir / "per_class_metrics.csv", metrics["per_class"])

    # Multiclass confusion matrix
    cm = metrics["confusion_matrix"]
    write_confusion_csv(out_dir / "confusion_matrix.csv", cm, loader.class_names)
    write_normalized_confusion_csv(out_dir / "confusion_matrix_normalized.csv", cm, loader.class_names)
    plot_confusion_matrix(out_dir / "confusion_matrix.png", cm, loader.class_names, title=f"{args.model.upper()} Confusion Matrix ({args.split})")
    plot_confusion_matrix(out_dir / "confusion_matrix_normalized.png", cm, loader.class_names, title=f"{args.model.upper()} Normalized Confusion Matrix ({args.split})")

    # Binary confusion matrix
    bin_cm = confusion_matrix(y_bin, y_pred_bin)
    write_confusion_csv(out_dir / "confusion_matrix_binary.csv", bin_cm.tolist(), ["Benign", "Attack"])
    plot_confusion_matrix(out_dir / "confusion_matrix_binary.png", bin_cm.tolist(), ["Benign", "Attack"], title=f"{args.model.upper()} Binary Confusion Matrix ({args.split})")

    # Precision-Recall curve
    write_pr_curve_csv(out_dir / "precision_recall_curve.csv", precision_vals, recall_vals, thr_vals)
    plot_pr_curve(out_dir / "precision_recall_curve.png", precision_vals, recall_vals, title=f"{args.model.upper()} Binary PR Curve ({args.split})")

    # Save predictions NPZ with authentic graph and edge IDs
    save_predictions_npz(
        out_dir / "predictions.npz",
        y_true_multiclass=y_multi,
        y_pred_multiclass=y_pred,
        y_true_binary=y_bin,
        y_pred_binary=y_pred_bin,
        y_proba_multiclass=y_proba,
        graph_id=graph_ids,
        edge_id=edge_ids,
        class_names=loader.class_names,
        benign_class_id=loader.benign_id,
        dataset=args.dataset.upper(),
        split=args.split,
    )

    # Run configuration metadata
    run_cfg = {
        "model_type": args.model,
        "dataset": args.dataset.upper(),
        "split": args.split,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_samples": n_samples,
        "n_graphs": len(np.unique(graph_ids)),
        "class_names": loader.class_names,
        "benign_id": loader.benign_id,
        "feature_count": X.shape[1],
        "inference_duration_seconds": inference_duration,
    }
    write_json(out_dir / "run_config.json", run_cfg)

    print(f"[INFO] Evaluation completed successfully. Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
