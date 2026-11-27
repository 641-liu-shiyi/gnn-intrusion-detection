"""Cross-dataset evaluation script.

This module implements a CLI that loads a checkpoint trained on a *source*
dataset (UNSW or CICIDS), transforms the *target* dataset's features to the
source preprocessing space, runs inference with the source model, and exports
standard evaluation artefacts (metrics, predictions, configuration).

The implementation follows the strict policy described in the project brief:

* No refitting of scalers - the original ``StandardScaler`` objects are used.
* Exact feature ordering is verified; missing source features raise an error.
* Binary label conversion is performed using the source dataset's ``Benign``
  class index.
* When ``--split test`` is requested, an additional ``--confirm-test`` flag is
  required to avoid accidental test-set leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.baselines.common import (
    normalize_dataset_name,
    resolve_default_data_root,
    resolve_feature_dir_name,
    resolve_split_name,
)
from src.baselines.data_loader import DataLoader
from src.baselines.eval_utils import (
    XGBModelWrapper,
    plot_confusion_matrix,
    save_predictions_npz,
    write_confusion_csv,
    write_csv,
    write_json,
)
from src.baselines.mlp_model import MLP
from src.eval.metrics import compute_metrics, compute_proba_metrics
from src.eval.preprocess_cross import inverse_transform_and_reorder
import __main__

setattr(__main__, "XGBModelWrapper", XGBModelWrapper)


def _load_scaler_and_features(
    dataset: str,
    data_root: Path,
    feature_dir_override: Path | None = None,
) -> tuple[Any, List[str], Path]:
    """Load ``scaler.joblib`` and ``feature_names.joblib`` for a dataset."""
    feature_dir = (
        feature_dir_override
        if feature_dir_override is not None
        else data_root / resolve_feature_dir_name(dataset)
    )
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found for dataset {dataset} at {feature_dir}")
    scaler = joblib.load(feature_dir / "scaler.joblib")
    feature_names: List[str] = joblib.load(feature_dir / "feature_names.joblib")
    return scaler, feature_names, feature_dir


def _load_class_names(dataset: str, data_root: Path, split_name: str | None = None) -> List[str]:
    resolved_split = split_name if split_name else resolve_split_name(dataset)
    class_path = data_root / resolved_split / "class_names.json"
    if not class_path.is_file():
        raise FileNotFoundError(f"class_names.json not found for dataset {dataset} at {class_path}")
    return json.loads(class_path.read_text(encoding="utf-8"))


def _load_model(model_type: str, checkpoint: Path, input_dim: int, n_classes: int):
    """Load a model checkpoint (``rf``, ``xgb``, or ``mlp``)."""
    if model_type in {"rf", "xgb"}:
        model = joblib.load(checkpoint)
        if not hasattr(model, "predict"):
            raise ValueError(f"Loaded {model_type} checkpoint does not have predict method")
        return model
    if model_type == "mlp":
        state = torch.load(checkpoint, map_location="cpu")
        hidden1 = state["net.0.weight"].shape[0] if "net.0.weight" in state else 64
        hidden2 = state["net.3.weight"].shape[0] if "net.3.weight" in state else 32
        out_dim = state["net.6.weight"].shape[0] if "net.6.weight" in state else n_classes
        model = MLP(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2, output_dim=out_dim)
        model.load_state_dict(state)
        model.eval()
        return model
    raise ValueError(f"Unsupported model type: {model_type}")


def _predict_mlp_batched(model: MLP, X: np.ndarray, batch_size: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Run CPU MLP inference in chunks to keep memory usage bounded on large splits."""
    proba_parts: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).float()
            logits = model(batch)
            proba_parts.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    y_proba = np.vstack(proba_parts)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation of a source model on a target dataset")
    parser.add_argument("--model", choices=["rf", "xgb", "mlp"], required=True, help="Model type to load")
    parser.add_argument("--source-dataset", choices=["unsw", "cicids", "cicids_episode"], required=True, help="Dataset on which the model was trained")
    parser.add_argument("--target-dataset", choices=["unsw", "cicids", "cicids_episode"], required=True, help="Dataset to evaluate on")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the source model checkpoint file")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Target split to evaluate (default: val)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where artefacts will be written")
    parser.add_argument("--data-root", type=Path, default=None, help="Shared root directory containing datasets (optional)")
    parser.add_argument("--source-data-root", type=Path, default=None, help="Root directory for the source dataset (optional)")
    parser.add_argument("--target-data-root", type=Path, default=None, help="Root directory for the target dataset (optional)")
    parser.add_argument("--source-split-name", type=str, default=None, help="Override graph directory name for source dataset")
    parser.add_argument("--target-split-name", type=str, default=None, help="Override graph directory name for target dataset")
    parser.add_argument("--source-feature-dir", type=Path, default=None, help="Override feature/scaler directory for source dataset")
    parser.add_argument("--target-feature-dir", type=Path, default=None, help="Override feature/scaler directory for target dataset")
    parser.add_argument("--confirm-test", action="store_true", help="Required flag to permit evaluation on the test split")
    args = parser.parse_args()

    source_norm = normalize_dataset_name(args.source_dataset)
    target_norm = normalize_dataset_name(args.target_dataset)
    if source_norm == target_norm:
        raise ValueError("Source and target datasets must differ for cross-dataset evaluation")
    if args.split == "test" and not args.confirm_test:
        raise ValueError("Evaluating on the test split requires explicit --confirm-test flag")

    repo_root = Path(__file__).resolve().parents[2]
    source_root = (
        args.source_data_root
        if args.source_data_root is not None
        else (args.data_root if args.data_root is not None else resolve_default_data_root(source_norm, repo_root=repo_root))
    )
    target_root = (
        args.target_data_root
        if args.target_data_root is not None
        else (args.data_root if args.data_root is not None else resolve_default_data_root(target_norm, repo_root=repo_root))
    )
    source_split_name = args.source_split_name if args.source_split_name else resolve_split_name(source_norm)
    target_split_name = args.target_split_name if args.target_split_name else resolve_split_name(target_norm)

    # ---------------------------------------------------------------------
    # Load source artefacts (scaler, feature names, class names)
    # ---------------------------------------------------------------------
    source_scaler, source_feature_names, source_feature_dir = _load_scaler_and_features(
        source_norm, source_root, feature_dir_override=args.source_feature_dir
    )
    source_class_names = _load_class_names(source_norm, source_root, split_name=source_split_name)
    source_n_classes = len(source_class_names)
    try:
        source_benign_id = source_class_names.index("Benign")
    except ValueError as exc:
        raise ValueError("'Benign' class not found in source class_names.json") from exc

    # ---------------------------------------------------------------------
    # Load target data
    # ---------------------------------------------------------------------
    loader = DataLoader(data_root=target_root, split_name=target_split_name)
    X_target, y_target_multi, y_target_binary, graph_ids, edge_ids = loader.load_split_with_ids(args.split)

    # Load target scaler/feature names
    target_scaler, target_feature_names, target_feature_dir = _load_scaler_and_features(
        target_norm, target_root, feature_dir_override=args.target_feature_dir
    )

    # ---------------------------------------------------------------------
    # Transform target features into the source preprocessing space
    # ---------------------------------------------------------------------
    X_source = inverse_transform_and_reorder(
        X_target.astype(np.float32),
        target_scaler,
        target_feature_names,
        source_scaler,
        source_feature_names,
    )

    # ---------------------------------------------------------------------
    # Load source model and perform inference
    # ---------------------------------------------------------------------
    model = _load_model(args.model, args.checkpoint, input_dim=len(source_feature_names), n_classes=source_n_classes)

    if args.model in {"rf", "xgb"}:
        y_pred_multi = model.predict(X_source)
        y_proba = model.predict_proba(X_source)
    else:  # mlp
        y_pred_multi, y_proba = _predict_mlp_batched(model, X_source)

    # Binary conversion based on source benign id
    y_pred_binary = (y_pred_multi != source_benign_id).astype(np.int64)
    prob_benign = y_proba[:, source_benign_id]
    binary_score = 1.0 - prob_benign
    # Construct a two-column probability matrix for binary metrics
    binary_proba = np.stack([prob_benign, binary_score], axis=1)

    # ---------------------------------------------------------------------
    # Compute metrics (binary only)
    # ---------------------------------------------------------------------
    binary_class_names = ["Benign", "Attack"]
    binary_metrics = compute_metrics(
        y_target_binary,
        y_pred_binary,
        class_names=binary_class_names,
        binary_true=y_target_binary,
        binary_pred=y_pred_binary,
        benign_id=0,
    )
    proba_metrics = compute_proba_metrics(
        y_target_binary,
        binary_proba,
        class_names=binary_class_names,
        benign_id=0,
        binary_true=y_target_binary,
    )
    metrics: Dict[str, Any] = {**binary_metrics, **proba_metrics}

    # ---------------------------------------------------------------------
    # Write output artefacts
    # ---------------------------------------------------------------------
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists():
        raise RuntimeError(f"Output directory {out_dir} already contains results; abort to avoid overwriting")

    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "metrics.csv", metrics)

    # Binary confusion matrix CSV and PNG
    binary_cm = confusion_matrix(y_target_binary, y_pred_binary)
    write_confusion_csv(
        out_dir / "confusion_matrix_binary.csv",
        binary_cm.tolist(),
        binary_class_names,
    )
    plot_confusion_matrix(
        out_dir / "confusion_matrix_binary.png",
        binary_cm.tolist(),
        binary_class_names,
        title="Binary confusion matrix",
    )

    # Save predictions using the shared utility
    save_predictions_npz(
        out_dir / "predictions.npz",
        y_true_multiclass=y_target_multi,
        y_pred_multiclass=y_pred_multi,
        y_true_binary=y_target_binary,
        y_pred_binary=y_pred_binary,
        y_proba_multiclass=binary_proba,
        graph_id=graph_ids,
        edge_id=edge_ids,
        class_names=binary_class_names,
        benign_class_id=0,
        dataset=target_norm,
        split=args.split,
    )

    # Run configuration JSON - captures the essential provenance information.
    run_cfg = {
        "model_type": args.model,
        "source_dataset": source_norm,
        "target_dataset": target_norm,
        "source_split_name": source_split_name,
        "target_split_name": target_split_name,
        "source_feature_dir": str(source_feature_dir),
        "target_feature_dir": str(target_feature_dir),
        "target_graph_dir": str(loader.graph_dir),
        "source_feature_count": len(source_feature_names),
        "target_feature_count": len(target_feature_names),
        "source_class_names": source_class_names,
        "source_benign_id": source_benign_id,
        "checkpoint_path": str(args.checkpoint),
        "split": args.split,
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "run_config.json", run_cfg)

    print("[INFO] Cross-dataset evaluation completed. Results written to", out_dir)


if __name__ == "__main__":
    main()
