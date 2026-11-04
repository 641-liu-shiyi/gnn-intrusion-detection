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
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import joblib
import torch
from sklearn.metrics import confusion_matrix

from src.baselines.data_loader import DataLoader
from src.eval.metrics import compute_metrics, compute_proba_metrics
from src.eval.preprocess_cross import inverse_transform_and_reorder
from src.baselines.eval_utils import (
    save_predictions_npz,
    write_json,
    write_csv,
    write_confusion_csv,
    plot_confusion_matrix,
    XGBModelWrapper,
)
from src.baselines.mlp_model import MLP
import __main__

setattr(__main__, "XGBModelWrapper", XGBModelWrapper)


def _load_scaler_and_features(dataset: str, data_root: Path) -> tuple[Any, List[str]]:
    """Load ``scaler.joblib`` and ``feature_names.joblib`` for a dataset.

    Parameters
    ----------
    dataset:
        ``"unsw"`` or ``"cicids"``.
    data_root:
        Path containing ``features`` or ``features_cicids`` subdirectories.
    """
    if dataset == "unsw":
        feature_dir = data_root / "features"
    else:
        feature_dir = data_root / "features_cicids"
    scaler = joblib.load(feature_dir / "scaler.joblib")
    feature_names: List[str] = joblib.load(feature_dir / "feature_names.joblib")
    return scaler, feature_names


def _load_class_names(dataset: str, data_root: Path) -> List[str]:
    class_path = data_root / f"graphs_{dataset}" / "class_names.json"
    if not class_path.is_file():
        raise FileNotFoundError(f"class_names.json not found for dataset {dataset} at {class_path}")
    return json.loads(class_path.read_text(encoding="utf-8"))


def _load_model(model_type: str, checkpoint: Path, input_dim: int, n_classes: int):
    """Load a model checkpoint.

    ``model_type`` must be one of ``rf``, ``xgb`` or ``mlp``.
    ``input_dim`` and ``n_classes`` are required for constructing an MLP.
    """
    if model_type == "rf":
        import joblib

        model = joblib.load(checkpoint)
        if not hasattr(model, "predict"):
            raise ValueError("Loaded RandomForest checkpoint does not have predict method")
        return model
    if model_type == "xgb":
        import joblib

        model = joblib.load(checkpoint)
        if not hasattr(model, "predict"):
            raise ValueError("Loaded XGBoost checkpoint does not have predict method")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation of a source model on a target dataset")
    parser.add_argument("--model", choices=["rf", "xgb", "mlp"], required=True, help="Model type to load")
    parser.add_argument("--source-dataset", choices=["unsw", "cicids"], required=True, help="Dataset on which the model was trained")
    parser.add_argument("--target-dataset", choices=["unsw", "cicids"], required=True, help="Dataset to evaluate on")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the source model checkpoint file")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Target split to evaluate (default: val)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where artefacts will be written")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing the graph datasets (optional)")
    parser.add_argument("--confirm-test", action="store_true", help="Required flag to permit evaluation on the test split")
    args = parser.parse_args()

    if args.source_dataset == args.target_dataset:
        raise ValueError("Source and target datasets must differ for cross-dataset evaluation")
    if args.split == "test" and not args.confirm_test:
        raise ValueError("Evaluating on the test split requires explicit --confirm-test flag")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root if args.data_root is not None else repo_root / "graph_data_bundle" / "data"

    # ---------------------------------------------------------------------
    # Load source artefacts (scaler, feature names, class names)
    # ---------------------------------------------------------------------
    source_scaler, source_feature_names = _load_scaler_and_features(args.source_dataset, data_root)
    source_class_names = _load_class_names(args.source_dataset, data_root)
    source_n_classes = len(source_class_names)
    try:
        source_benign_id = source_class_names.index("Benign")
    except ValueError as exc:
        raise ValueError("'Benign' class not found in source class_names.json") from exc

    # ---------------------------------------------------------------------
    # Load target data
    # ---------------------------------------------------------------------
    loader = DataLoader(data_root=data_root, split_name=f"graphs_{args.target_dataset}")
    if args.split == "val":
        X_target, y_target_multi, y_target_binary, graph_ids, edge_ids = loader.load_split_with_ids("val")
    else:
        X_target, y_target_multi, y_target_binary, graph_ids, edge_ids = loader.load_split_with_ids("test")

    # Load target scaler/feature names
    target_scaler, target_feature_names = _load_scaler_and_features(args.target_dataset, data_root)

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
        with torch.no_grad():
            logits = model(torch.from_numpy(X_source).float())
            y_proba = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            y_pred_multi = np.argmax(y_proba, axis=1)

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
    # Merge metric dictionaries for convenience
    metrics: Dict[str, Any] = {**binary_metrics, **proba_metrics}

    # ---------------------------------------------------------------------
    # Write output artefacts
    # ---------------------------------------------------------------------
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists():
        raise RuntimeError(f"Output directory {out_dir} already contains results; abort to avoid overwriting")

    write_json(out_dir / "metrics.json", metrics)
    # Simple CSV version - write flat dict (one row)
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

    # Save predictions using the shared utility - note we store source class names
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
        dataset=args.target_dataset,
        split=args.split,
    )

    # Run configuration JSON - captures the essential provenance information.
    run_cfg = {
        "model_type": args.model,
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
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
