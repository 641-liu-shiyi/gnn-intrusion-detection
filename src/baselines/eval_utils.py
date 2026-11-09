"""Utility helpers for writing evaluation artefacts for baseline scripts.

The original baseline implementations wrote only ``metrics.json``/``metrics.csv``
and a single confusion-matrix CSV.  Part 2 requires richer artefacts - per-class
CSV, normalized confusion matrix (CSV + PNG), binary confusion matrix (CSV + PNG),
precision-recall curve (CSV + PNG) and the ``predictions.npz`` export used by the
reporting tools.

All functions below are deliberately lightweight and have no side-effects
outside the supplied ``out_dir``.  They are imported by the three baseline
training scripts (RF, XGBoost, MLP) to avoid code duplication.
"""

from __future__ import annotations

import json
import csv
import pathlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import confusion_matrix

# Matplotlib is optional - the tests run in a headless environment.  We enforce
# the non-interactive Agg backend to avoid GUI requirements.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    "write_json",
    "write_csv",
    "write_per_class_csv",
    "write_confusion_csv",
    "write_normalized_confusion_csv",
    "plot_confusion_matrix",
    "write_pr_curve_csv",
    "plot_pr_curve",
    "save_predictions_npz",
    "align_probabilities",
    "XGBModelWrapper",
]


def write_json(path: Path, data: Any) -> None:
    """Write *data* as pretty-printed JSON to *path* (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def write_csv(path: Path, rows: Dict[str, Any]) -> None:
    """Write a simple ``metric,value`` CSV used for flattened metric tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, val in rows.items():
            writer.writerow([key, val])


def write_per_class_csv(path: Path, per_class: Dict[str, Dict[str, Any]]) -> None:
    """Write per-class metrics (precision, recall, f1, support) to CSV.

    The columns are: ``class,precision,recall,f1,support``.  Values are formatted
    as floating-point numbers with four decimal places for consistency with the
    existing scalar tables.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for class_name, metrics in per_class.items():
            writer.writerow(
                [
                    class_name,
                    f"{metrics.get('precision', 0.0):.4f}",
                    f"{metrics.get('recall', 0.0):.4f}",
                    f"{metrics.get('f1', 0.0):.4f}",
                    metrics.get("support", 0),
                ]
            )


def write_confusion_csv(path: Path, cm: List[List[int]], class_names: List[str]) -> None:
    """Write a raw confusion matrix CSV with class names as headers/row labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, cm):
            writer.writerow([name] + list(row))


def write_normalized_confusion_csv(
    path: Path, cm: List[List[int]], class_names: List[str]
) -> None:
    """Write a row-normalised confusion matrix (each row sums to 1.0)."""
    arr = np.array(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    # Avoid division by zero - rows with sum 0 remain zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums == 0, 0.0, arr / row_sums)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, norm):
            writer.writerow([name] + [f"{v:.4f}" for v in row])


def plot_confusion_matrix(
    out_path: Path, cm: List[List[int]], class_names: List[str], title: str = "Confusion matrix"
) -> None:
    """Render a confusion matrix as a heat-map PNG.

    ``out_path`` should include the ``.png`` suffix. The function creates parent
    directories as needed. The input ``cm`` may be a plain ``list`` of ``list``
    values (as produced by ``sklearn.metrics.confusion_matrix``).  ``imshow``
    expects a NumPy array, so we explicitly convert ``cm`` to an ``ndarray`` to
    avoid attribute errors when computing the colour-threshold.
    """
    # Convert to NumPy array for safe numeric operations and for ``imshow``.
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = im.norm(cm_arr.max()) / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                f"{cm_arr[i][j]}",
                ha="center",
                va="center",
                color="white" if cm_arr[i][j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_pr_curve_csv(
    out_path: Path,
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    """Write precision-recall curve data to CSV.

    The ``thresholds`` array is one element shorter than ``precisions``/``recalls``;
    we pad it with ``nan`` for the final row so that each line contains a trio of
    values.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "precision", "recall"])
        # Pad thresholds to match length of precisions/recalls
        padded_thr = np.append(thresholds, np.nan)
        for thr, prec, rec in zip(padded_thr, precisions, recalls):
            writer.writerow([thr, f"{prec:.6f}", f"{rec:.6f}"])


def plot_pr_curve(
    out_path: Path,
    precisions: np.ndarray,
    recalls: np.ndarray,
    title: str = "Precision-Recall curve",
) -> None:
    """Save a PNG of the PR curve (recall on X-axis, precision on Y-axis)."""
    fig, ax = plt.subplots()
    ax.step(recalls, precisions, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_predictions_npz(
    out_path: Path,
    *,
    y_true_multiclass: np.ndarray,
    y_pred_multiclass: np.ndarray,
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_proba_multiclass: np.ndarray | None,
    graph_id: np.ndarray,
    edge_id: np.ndarray,
    class_names: List[str],
    benign_class_id: int,
    dataset: str,
    split: str,
) -> None:
    """Save all prediction related arrays to a compressed ``.npz`` file.

    The function deliberately does **not** store any raw feature data - only
    identifiers and model outputs required for downstream analysis.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        y_true_multiclass=y_true_multiclass,
        y_pred_multiclass=y_pred_multiclass,
        y_true_binary=y_true_binary,
        y_pred_binary=y_pred_binary,
        y_proba_multiclass=y_proba_multiclass,
        graph_id=graph_id,
        edge_id=edge_id,
        # Store class names as a Unicode string array to avoid object pickling.
        class_names=np.array(class_names, dtype=str),
        benign_class_id=np.int64(benign_class_id),
        dataset=dataset,
        split=split,
    )


def align_probabilities(proba: np.ndarray, model_classes: np.ndarray, num_classes: int) -> np.ndarray:
    """Align model predicted probabilities so that column j corresponds to class j (0 <= j < num_classes).

    If the training set had a subset of classes, model.predict_proba returns only columns
    for model.classes_. This aligns them into a matrix of shape (n_samples, num_classes).
    """
    if proba.shape[1] == num_classes and np.array_equal(model_classes, np.arange(num_classes)):
        return proba
    aligned = np.zeros((proba.shape[0], num_classes), dtype=proba.dtype)
    for col_idx, cls_idx in enumerate(model_classes):
        cls_int = int(cls_idx)
        if 0 <= cls_int < num_classes:
            aligned[:, cls_int] = proba[:, col_idx]
    return aligned


class XGBModelWrapper:
    """Wrapper around XGBClassifier that handles label encoding for missing training classes."""

    def __init__(self, clf: Any, label_encoder: Any, num_classes: int):
        self.clf = clf
        self.label_encoder = label_encoder
        self.num_classes = num_classes
        self.classes_ = np.arange(num_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.clf.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.clf.predict_proba(X)
        return align_probabilities(proba, self.label_encoder.classes_, self.num_classes)

    def __getattr__(self, name: str) -> Any:
        if (name.startswith("__") and name.endswith("__")) or "clf" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.clf, name)

