"""Utilities for importing external GNN (E-GraphSAGE) experiment artefacts.

The baseline training scripts already produce a well-defined set of artefacts:

* ``metrics.json`` - a dictionary containing scalar validation metrics.
* ``run_config.json`` - a JSON blob with the configuration used for the run.
* ``predictions.npz`` - a compressed NumPy archive with per-flow predictions and
  identifiers (``graph_id``/``edge_id``).  The schema mirrors the one used by the
  baseline helpers in :mod:`src.baselines.eval_utils`.

Teammates providing the result of an external GNN model (e.g. E-GraphSAGE) are
required to follow this schema exactly.  This module validates the supplied
files, loads them into memory and raises clear ``ValueError`` exceptions when
any requirement is violated.

Only import-time validation is performed - the heavy-weight metric recomputation
is delegated to :func:`src.eval.metrics.compute_metrics` and
``compute_proba_metrics`` by the caller when needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# The baseline ``report`` module defines the set of required scalar metric keys.
# We reuse the same set for external artefacts to ensure compatibility.
REQUIRED_METRIC_KEYS = {"accuracy", "macro_f1", "weighted_f1", "binary_f1"}


class GNNImportError(ValueError):
    """Raised when an external GNN artefact does not meet the required schema."""


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise GNNImportError(f"Required JSON file missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive programming
        raise GNNImportError(f"Failed to parse JSON at {path}: {exc}") from exc


def _validate_metrics(metrics: Mapping[str, Any]) -> Mapping[str, float]:
    """Validate that ``metrics`` contains the scalar keys used by the report.

    Returns a dict with the values cast to ``float`` for downstream convenience.
    """
    missing = REQUIRED_METRIC_KEYS - metrics.keys()
    if missing:
        raise GNNImportError(f"Metrics JSON missing required keys: {sorted(missing)}")
    out: dict[str, float] = {}
    for key in REQUIRED_METRIC_KEYS:
        val = metrics[key]
        if not isinstance(val, (int, float)):
            raise GNNImportError(f"Metric {key!r} must be numeric, got {type(val)}")
        out[key] = float(val)
    return out


def _validate_predictions(npz: np.lib.npyio.NpzFile) -> Mapping[str, Any]:
    """Validate the ``predictions.npz`` schema.

    Required fields (as written by :func:`src.baselines.eval_utils.save_predictions_npz`):

    * ``y_true_multiclass`` - ``(n,)`` integer array
    * ``y_pred_multiclass`` - ``(n,)`` integer array
    * ``y_true_binary`` - ``(n,)`` integer array (0/1)
    * ``y_pred_binary`` - ``(n,)`` integer array (0/1)
    * ``y_proba_multiclass`` - ``(n, n_classes)`` float array **optional**
    * ``graph_id`` - ``(n,)`` integer array
    * ``edge_id`` - ``(n,)`` integer array
    * ``class_names`` - 1-D array of unicode strings (length = n_classes)
    * ``benign_class_id`` - scalar integer
    * ``dataset`` - string scalar
    * ``split`` - string scalar (e.g. ``"val"``)
    """
    required = {
        "y_true_multiclass",
        "y_pred_multiclass",
        "y_true_binary",
        "y_pred_binary",
        "graph_id",
        "edge_id",
        "class_names",
        "benign_class_id",
        "dataset",
        "split",
    }
    missing = required - set(npz.files)
    if missing:
        raise GNNImportError(f"predictions.npz missing required arrays: {sorted(missing)}")

    y_true_multi = npz["y_true_multiclass"]
    y_pred_multi = npz["y_pred_multiclass"]
    y_true_bin = npz["y_true_binary"]
    y_pred_bin = npz["y_pred_binary"]
    graph_id = npz["graph_id"]
    edge_id = npz["edge_id"]
    class_names = npz["class_names"]
    benign_id = npz["benign_class_id"]
    dataset = npz["dataset"]
    split = npz["split"]

    n = y_true_multi.shape[0]
    # Ensure all per-flow arrays have the same length.
    for name, arr in [
        ("y_pred_multiclass", y_pred_multi),
        ("y_true_binary", y_true_bin),
        ("y_pred_binary", y_pred_bin),
        ("graph_id", graph_id),
        ("edge_id", edge_id),
    ]:
        if arr.shape != (n,):
            raise GNNImportError(f"Array {name} has shape {arr.shape}, expected ({n},)")

    n_classes = class_names.shape[0]
    if "y_proba_multiclass" in npz.files:
        proba = npz["y_proba_multiclass"]
        if proba.shape != (n, n_classes):
            raise GNNImportError(
                f"y_proba_multiclass shape {proba.shape} incompatible with n={n} and n_classes={n_classes}"
            )
        if not np.all(np.isfinite(proba)):
            raise GNNImportError("y_proba_multiclass contains non-finite values")
        if not np.all((proba >= 0) & (proba <= 1)):
            raise GNNImportError("y_proba_multiclass contains values outside [0, 1]")

    # ``class_names`` must be a one-dimensional string array.
    if class_names.ndim != 1:
        raise GNNImportError("class_names must be a one-dimensional array")
    if not np.issubdtype(class_names.dtype, np.str_):
        raise GNNImportError("class_names array must contain strings")

    # ``benign_class_id`` should be scalar integer within range.  ``np.savez``
    # sometimes stores a Python ``int`` as a zero-dimensional ``ndarray``.  We
    # accept both a true scalar and a 0-d ``ndarray`` with an integer dtype.
    if isinstance(benign_id, np.ndarray):
        if benign_id.shape != ():
            raise GNNImportError("benign_class_id must be a scalar integer (got array with shape {benign_id.shape})")
        if not np.issubdtype(benign_id.dtype, np.integer):
            raise GNNImportError("benign_class_id ndarray must contain an integer value")
        benign_id_int = int(benign_id)
    else:
        if not isinstance(benign_id, (int, np.integer)):
            raise GNNImportError("benign_class_id must be an integer scalar")
        benign_id_int = int(benign_id)
    if not (0 <= benign_id_int < n_classes):
        raise GNNImportError(
            f"benign_class_id {benign_id_int} out of range for {n_classes} classes"
        )

    dataset_str = str(dataset)
    split_str = str(split)
    if not dataset_str:
        raise GNNImportError("dataset field in predictions.npz is empty")
    if not split_str:
        raise GNNImportError("split field in predictions.npz is empty")

    out: dict[str, Any] = {
        "y_true_multiclass": y_true_multi,
        "y_pred_multiclass": y_pred_multi,
        "y_true_binary": y_true_bin,
        "y_pred_binary": y_pred_bin,
        "graph_id": graph_id,
        "edge_id": edge_id,
        "class_names": class_names,
        "benign_class_id": benign_id_int,
        "dataset": dataset_str,
        "split": split_str,
    }
    if "y_proba_multiclass" in npz.files:
        out["y_proba_multiclass"] = npz["y_proba_multiclass"]
    return out


def load_gnn_artifacts(base_dir: Path) -> dict[str, Any]:
    """Load and validate the artefacts produced by an external GNN experiment.

    The directory must contain ``metrics.json``, ``run_config.json`` and
    ``predictions.npz``.  The function returns a mapping with three top-level
    keys: ``"metrics"``, ``"run_config"`` and ``"predictions"``.
    """
    base_dir = Path(base_dir)
    metrics_path = base_dir / "metrics.json"
    run_cfg_path = base_dir / "run_config.json"
    pred_path = base_dir / "predictions.npz"

    metrics_raw = _load_json(metrics_path)
    metrics = _validate_metrics(metrics_raw)

    run_cfg = _load_json(run_cfg_path)

    try:
        with np.load(pred_path, allow_pickle=False) as npz:
            predictions = _validate_predictions(npz)
    except Exception as exc:  # pragma: no cover - defensive
        raise GNNImportError(f"Failed to load predictions.npz: {exc}") from exc

    return {"metrics": metrics, "run_config": run_cfg, "predictions": predictions}


__all__ = ["load_gnn_artifacts", "GNNImportError"]
