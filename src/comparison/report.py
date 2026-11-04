"""Utility functions for comparing baseline experiment results.

The module is intentionally lightweight - it does not depend on any heavy
visualisation libraries.  It reads the ``run_config.json`` and ``metrics.json``
produced by each baseline script and extracts the most important scalar metrics
required for the checkpoint report (accuracy, macro F1, weighted F1, binary F1).

Two public helpers are provided:

``load_results(base_dir: Path)``
    Load ``metrics.json`` from the given directory and return the parsed dict.

``compare_metrics(results: dict[str, dict])``
    Accept a mapping ``model_name -> metrics_dict`` and return a markdown table
    string that lists the selected metrics side-by-side.  The function raises a
    ``ValueError`` if any of the supplied result dictionaries miss a required
    field, ensuring that the comparison is always performed on compatible data.

These helpers are sufficient for the unit tests and for teammates to generate a
quick textual comparison.  More sophisticated reporting (CSV, LaTeX, plots) can
be built on top of them later without changing the core logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Any, Iterable

from src.eval.metrics import compute_metrics, compute_proba_metrics
import numpy as np

REQUIRED_KEYS = {"accuracy", "macro_f1", "weighted_f1", "binary_f1"}


def load_results(base_dir: Path) -> Dict[str, Any]:
    """Load ``metrics.json`` from *base_dir*.

    Parameters
    ----------
    base_dir: Path
        Directory that contains the baseline output (e.g. ``results/rf_unsw``).

    Returns
    -------
    dict
        Parsed JSON content.
    """
    metrics_path = base_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "validation" in data and isinstance(data["validation"], dict):
        return data["validation"]
    return data


def _extract_scalar(metrics: Mapping[str, Any]) -> Mapping[str, float]:
    """Return a dict with only the scalar metrics we report.

    The function validates that all ``REQUIRED_KEYS`` are present and are numbers.
    """
    if isinstance(metrics, dict) and "validation" in metrics and isinstance(metrics["validation"], dict):
        metrics = metrics["validation"]
    missing = REQUIRED_KEYS - metrics.keys()
    if missing:
        raise ValueError(f"Missing required metric keys: {sorted(missing)}")
    out: dict[str, float] = {}
    for key in REQUIRED_KEYS:
        val = metrics[key]
        if not isinstance(val, (int, float)):
            raise TypeError(f"Metric {key!r} is not numeric: {type(val)}")
        out[key] = float(val)
    return out


def compare_metrics(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Create a Markdown table comparing baseline metrics.

    The table follows the standard GitHub-flavoured Markdown format:
    ``| Model | Accuracy | Macro F1 | Weighted F1 | Binary F1 |``
    ``|---|---|---|---|---|``
    ``| RF    | 0.9551   | 0.3755  | 0.9536      | 0.7977   |``
    """
    header = "| Model | Accuracy | Macro F1 | Weighted F1 | Binary F1 |"
    separator = "|---|---|---|---|---|"
    lines = [header, separator]
    for model_name, raw_metrics in results.items():
        scalar = _extract_scalar(raw_metrics)
        line = (
            f"| {model_name} "
            f"| {scalar['accuracy']:.4f} "
            f"| {scalar['macro_f1']:.4f} "
            f"| {scalar['weighted_f1']:.4f} "
            f"| {scalar['binary_f1']:.4f} |"
        )
        lines.append(line)
    return "\n".join(lines)
def compare_metrics_latex(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Create a LaTeX table comparing baseline metrics.

    The table uses the ``IEEEtran`` compatible ``tabular`` environment and mirrors
    the column order of :func:`compare_metrics`.  All numeric values are formatted
    with four decimal places.
    """
    # Header row - LaTeX style (IEEEtran compatible)
    header = (
        "\\begin{tabular}{lrrrr}\n"
        "  \\toprule\n"
        "  Model & Accuracy & Macro\\,F1 & Weighted\\,F1 & Binary\\,F1 \\\\ \n"
        "  \\midrule"
    )
    lines = [header]
    for model_name, raw_metrics in results.items():
        scalar = _extract_scalar(raw_metrics)
        line = (
            f"  {model_name} & "
            f"{scalar['accuracy']:.4f} & "
            f"{scalar['macro_f1']:.4f} & "
            f"{scalar['weighted_f1']:.4f} & "
            f"{scalar['binary_f1']:.4f} \\\\"
        )
        lines.append(line)
    # Footer - bottom rule and end of tabular environment
    lines.append("  \\bottomrule\n\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Additional comparison utilities - aggregate vs. strict paired comparison
# ---------------------------------------------------------------------------

def _ensure_same_keys(models: Mapping[str, Mapping[str, Any]]) -> None:
    """Verify that all models provide predictions with identical identifier keys.

    The function raises :class:`ValueError` if any mismatch is detected.  It is
    used by :func:`compare_strict` to guarantee a fair paired comparison.
    """
    key_sets: list[set[tuple[int, int]]] = []
    for name, data in models.items():
        preds = data.get("predictions")
        if preds is None:
            raise ValueError(f"Model {name!r} missing 'predictions' for strict comparison")
        graph_ids = preds["graph_id"]
        edge_ids = preds["edge_id"]
        if graph_ids.shape != edge_ids.shape:
            raise ValueError(f"Model {name!r} graph_id/edge_id shape mismatch")
        keys = set(zip(map(int, graph_ids), map(int, edge_ids)))
        key_sets.append(keys)
    # All key sets must be identical.
    first = key_sets[0]
    for i, ks in enumerate(key_sets[1:], start=1):
        if ks != first:
            raise ValueError(
                f"Prediction identifier sets differ between models (model index 0 vs {i})"
            )


def _recompute_metrics_from_predictions(preds: Mapping[str, Any]) -> Mapping[str, float]:
    """Re-compute scalar metrics from a ``predictions`` dictionary.

    The function extracts the required arrays and forwards them to
    :func:`src.eval.metrics.compute_metrics`.  It returns a dict containing the
    scalar metrics defined in ``REQUIRED_KEYS`` (plus optional binary average
    precision if probability scores are available).
    """
    y_true = preds["y_true_multiclass"]
    y_pred = preds["y_pred_multiclass"]
    class_names = list(preds["class_names"].astype(str))
    benign_id = preds["benign_class_id"]
    # Compute standard multiclass metrics.
    met = compute_metrics(y_true, y_pred, class_names, benign_id=benign_id)
    # Extract the subset required for the table.
    out: dict[str, float] = {k: float(met[k]) for k in REQUIRED_KEYS if k in met}
    # If probability scores are present, add binary average precision.
    if "y_proba_multiclass" in preds:
        y_score = preds["y_proba_multiclass"]
        proba_met = compute_proba_metrics(
            y_true,
            y_score,
            class_names,
            benign_id=benign_id,
        )
        if "binary_average_precision" in proba_met:
            out["binary_average_precision"] = float(proba_met["binary_average_precision"])
    return out


def compare_aggregate(models: Mapping[str, Mapping[str, Any]]) -> str:
    """Generate a markdown table comparing *aggregate* metrics for multiple models.

    ``models`` may be a mapping of model name -> ``metrics`` dict (as loaded by
    :func:`load_results`) *or* a mapping of model name -> the full artefact dict
    returned by :func:`src.comparison.gnn_importer.load_gnn_artifacts`.  In the
    latter case the ``metrics`` sub-dictionary is extracted automatically.
    """
    # Normalise input - ensure we have a dict of raw metric mappings.
    normalized: dict[str, Mapping[str, Any]] = {}
    for name, data in models.items():
        if isinstance(data, dict) and "metrics" in data:
            normalized[name] = data["metrics"]
        else:
            normalized[name] = data
    return compare_metrics(normalized)


def compare_strict(models: Mapping[str, Mapping[str, Any]]) -> str:
    """Perform a *strict paired* comparison.

    The function validates that all supplied models share the same identifier set
    (graph/edge keys) and that the ground-truth labels are identical.  It then
    recomputes the scalar metrics from the per-flow predictions (ensuring a fair
    comparison) and returns a markdown table identical in format to
    :func:`compare_aggregate`.
    """
    # Verify identifier compatibility.
    _ensure_same_keys(models)
    # Verify that true labels match across models.
    # Use the first model as reference.
    ref_name, ref_data = next(iter(models.items()))
    ref_true = ref_data["predictions"]["y_true_multiclass"]
    for name, data in models.items():
        if name == ref_name:
            continue
        other_true = data["predictions"]["y_true_multiclass"]
        if not np.array_equal(ref_true, other_true):
            raise ValueError(f"True labels differ between {ref_name!r} and {name!r}")
    # Re-compute metrics for each model.
    recomputed: dict[str, Mapping[str, Any]] = {}
    for name, data in models.items():
        recomputed[name] = _recompute_metrics_from_predictions(data["predictions"])
    return compare_metrics(recomputed)

__all__ = [
    "load_results",
    "compare_metrics",
    "compare_metrics_latex",
    "compare_aggregate",
    "compare_strict",
]
