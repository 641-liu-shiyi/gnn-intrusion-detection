"""Common utilities shared across baseline training scripts.

The original ``train_rf.py`` and ``train_xgb.py`` already implement a ``--dataset``
argument that selects the underlying graph directory (``graphs_unsw`` or
``graphs_cicids``).  The MLP baseline was missing this feature.  To keep the
behaviour consistent we expose a tiny helper function that resolves the appropriate
``split_name`` for the :class:`src.baselines.data_loader.DataLoader` based on the
dataset name provided by the CLI.

Only a very small amount of logic is required - the function validates the input
value, constructs the directory name and returns it.  It is deliberately kept
dependency-free so that the baseline scripts can import it without any circular
imports.
"""

from __future__ import annotations


def resolve_split_name(dataset: str) -> str:
    """Return the sub-directory name used inside ``graph_data_bundle/data``.

    Parameters
    ----------
    dataset: str
        Either ``"unsw"`` or ``"cicids"`` - the user-facing names used by the
        command-line interface.  The function is case-insensitive.

    Returns
    -------
    str
        The directory name that contains the NPZ graph files for the chosen
        dataset (e.g. ``"graphs_unsw"``).
    """
    normalized = dataset.lower()
    if normalized == "unsw":
        return "graphs_unsw"
    if normalized == "cicids":
        return "graphs_cicids"
    raise ValueError(f"Unsupported dataset: {dataset!r}. Choose 'unsw' or 'cicids'.")


__all__ = ["resolve_split_name"]
