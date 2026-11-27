"""Common utilities shared across baseline training and evaluation scripts.

Provides configurable dataset-to-directory resolution for both UNSW-NB15 and the
corrected episode-aware CIC-IDS2018 split (``graphs_cicids_episode`` /
``features_cicids_episode``) without modifying UNSW behavior or falling back to
the obsolete chronological CICIDS split.
"""

from __future__ import annotations

from pathlib import Path


def normalize_dataset_name(dataset: str) -> str:
    """Normalize user-facing dataset CLI names to ``'unsw'`` or ``'cicids'``."""
    normalized = dataset.strip().lower()
    if normalized == "unsw":
        return "unsw"
    if normalized in {"cicids", "cicids_episode"}:
        return "cicids"
    raise ValueError(
        f"Unsupported dataset: {dataset!r}. Choose 'unsw', 'cicids', or 'cicids_episode'."
    )


def resolve_split_name(dataset: str) -> str:
    """Return the graph sub-directory name for ``dataset``."""
    normalized = normalize_dataset_name(dataset)
    if normalized == "unsw":
        return "graphs_unsw"
    return "graphs_cicids_episode"


def resolve_feature_dir_name(dataset: str) -> str:
    """Return the feature/scaler sub-directory name for ``dataset``."""
    normalized = normalize_dataset_name(dataset)
    if normalized == "unsw":
        return "features"
    return "features_cicids_episode"


def resolve_default_data_root(dataset: str, repo_root: Path | None = None) -> Path:
    """Resolve the default data root for ``dataset`` when ``--data-root`` is omitted."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    normalized = normalize_dataset_name(dataset)
    if normalized == "cicids":
        return root / "data"
    if (root / "data" / "graphs_unsw").is_dir():
        return root / "data"
    return root / "graph_data_bundle" / "data"


def resolve_data_root_for_split(split_name: str, repo_root: Path | None = None) -> Path:
    """Resolve the default data root for an explicit graph ``split_name``."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    if split_name == "graphs_cicids_episode":
        return root / "data"
    if (root / "data" / split_name).is_dir():
        return root / "data"
    return root / "graph_data_bundle" / "data"


__all__ = [
    "normalize_dataset_name",
    "resolve_split_name",
    "resolve_feature_dir_name",
    "resolve_default_data_root",
    "resolve_data_root_for_split",
]
