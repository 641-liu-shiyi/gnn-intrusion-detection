"""Reusable evaluation utilities.

The functions in this module are deliberately independent of any model or
training code - they operate purely on ``y_true`` / ``y_pred`` arrays and a list
of class names.  This makes them suitable for later XGBoost, MLP or GNN
baselines.

All metrics follow ``sklearn`` conventions with ``zero_division=0`` to avoid
division-by-zero warnings.  The return value is a plain ``dict`` that can be
JSON-serialised without further conversion.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    average_precision_score,
)

__all__ = ["compute_metrics", "compute_proba_metrics"]


def _binary_from_multiclass(
    y_multi: np.ndarray, *, benign_id: int
) -> np.ndarray:
    """Convert multiclass ground-truth / predictions to binary attack labels.

    The binary label is ``0`` for the *Benign* class and ``1`` for any attack.
    ``benign_id`` is the integer index of the "Benign" entry in the class
    mapping (derived from ``class_names.json``).
    """
    return (y_multi != benign_id).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    *,
    binary_true: Optional[np.ndarray] = None,
    binary_pred: Optional[np.ndarray] = None,
    benign_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate a comprehensive set of evaluation metrics.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of shape ``(n_samples,)`` containing integer class IDs.
    class_names:
        Ordered list of class names - the position in the list corresponds to the
        integer label used in ``y_true``/``y_pred``.
    binary_true, binary_pred:
        Optional pre-computed binary attack labels (``0`` for benign, ``1`` for any
        attack).  If omitted, they are derived from ``y_true``/``y_pred`` using
        ``benign_id``.
    benign_id:
        Index of the "Benign" class.  Required when ``binary_true``/``binary_pred``
        are not supplied.

    Returns
    -------
    dict
        A JSON-serialisable dictionary containing:

        * overall accuracy
        * macro / weighted precision, recall, F1
        * per-class metrics (precision, recall, F1, support)
        * confusion matrix (as a plain list of rows)
        * binary-attack precision, recall, F1
        * number of evaluated samples (``n_samples``)
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    n_classes = len(class_names)
    labels = list(range(n_classes))

    # --------------------------
    # Multiclass metrics
    # --------------------------
    accuracy = accuracy_score(y_true, y_pred)

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    macro_precision = float(np.mean(precision_arr))
    macro_recall = float(np.mean(recall_arr))
    macro_f1 = float(np.mean(f1_arr))

    # Weighted averages - use support as the weighting factor
    # Weighted averages - use sklearn's built-in weighted calculations for numerical stability.
    weighted_precision = float(
        precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    weighted_recall = float(
        recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    # NOTE: Previously the code attempted to recompute ``weighted_f1`` using a manual
    # weighted average of per-class F1 scores (``np.sum(f1_arr * support_arr) / total_support``).
    # However ``total_support`` was never defined, leading to a ``NameError`` during metric
    # calculation. The scikit-learn ``f1_score`` call already provides a numerically-stable
    # weighted F1, so the redundant computation has been removed.

    # Per-class dictionary (preserve ordering of class_names)
    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision_arr[idx]),
            "recall": float(recall_arr[idx]),
            "f1": float(f1_arr[idx]),
            "support": int(support_arr[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --------------------------
    # Binary attack metrics
    # --------------------------
    if binary_true is None or binary_pred is None:
        if benign_id is None:
            raise ValueError(
                "When binary_true/binary_pred are not provided, benign_id must be supplied"
            )
        binary_true = _binary_from_multiclass(y_true, benign_id=benign_id)
        binary_pred = _binary_from_multiclass(y_pred, benign_id=benign_id)

    binary_precision = precision_score(binary_true, binary_pred, zero_division=0)
    binary_recall = recall_score(binary_true, binary_pred, zero_division=0)
    binary_f1 = f1_score(binary_true, binary_pred, zero_division=0)

    return {
        "n_samples": int(y_true.shape[0]),
        "accuracy": float(accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "binary_precision": float(binary_precision),
        "binary_recall": float(binary_recall),
        "binary_f1": float(binary_f1),
        "class_names": class_names,
    }


def compute_proba_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    *,
    benign_id: int,
    binary_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute probability-based metrics (average precision) for multiclass and binary.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class indices.
    y_score : np.ndarray of shape (n_samples, n_classes)
        Predicted class probabilities (should sum to 1 for each sample).  The order of
        columns must match ``class_names``.
    class_names : List[str]
        Ordered list of class names corresponding to columns of ``y_score``.
    benign_id : int
        Index of the benign class - used to derive binary attack scores.
    binary_true : Optional[np.ndarray]
        Pre-computed binary ground-truth (0 for benign, 1 for attack).  If omitted the
        function derives it from ``y_true`` and ``benign_id``.

    Returns
    -------
    dict
        Mapping containing:
        * ``binary_average_precision`` - AP for the attack vs benign binary task.
        * ``per_class_average_precision`` - dict mapping class name to AP (one-vs-rest).
    """

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same number of samples")
    n_classes = len(class_names)
    if y_score.shape[1] != n_classes:
        raise ValueError("Number of columns in y_score must match length of class_names")

    # Binary ground truth
    if binary_true is None:
        binary_true = _binary_from_multiclass(y_true, benign_id=benign_id)

    # Binary scores - probability of attack (any non-benign class).  We use
    # ``1 - prob[benign]`` which is mathematically equivalent to ``sum prob[others]``.
    prob_benign = y_score[:, benign_id]
    binary_score = 1.0 - prob_benign
    # Guard against numerical issues (e.g. probabilities slightly outside [0,1])
    binary_score = np.clip(binary_score, 0.0, 1.0)

    binary_average_precision = float(
        average_precision_score(binary_true, binary_score, pos_label=1)
    )

    # Per-class one-vs-rest AP
    per_class_ap: Dict[str, float] = {}
    for idx, name in enumerate(class_names):
        # Binarise true labels for the current class
        binary_y = (y_true == idx).astype(int)
        # Use the column probabilities directly - they already represent P(class)
        class_score = y_score[:, idx]
        # average_precision_score returns 0 when there are no positive samples; we keep that.
        ap = float(average_precision_score(binary_y, class_score))
        per_class_ap[name] = ap

    return {
        "binary_average_precision": binary_average_precision,
        "per_class_average_precision": per_class_ap,
    }
