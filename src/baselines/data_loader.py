"""Streaming data loader for the UNSW graph dataset.

The loader reads the pre-computed ``.npz`` graph files located under
``graph_data_bundle/data/graphs_unsw``.  It extracts the flow-level feature matrix
(``edge_attr``) and the multiclass attack label (``edge_attack``) while performing
light-weight validation.  Validation includes:

* ``edge_attr`` must have 39 columns (the expected number of predictive features).
* ``edge_label`` (binary ground truth) must be consistent with the derived binary
  label ``(edge_attack != benign_id)``.

Training data can be optionally subsampled in a **deterministic, class-aware**
fashion to respect a user-specified maximum number of training samples.  The
sub-sampling policy follows the specification from the project description:

* Classes with <= 500 examples are kept in full when the overall budget permits.
* Remaining budget is allocated proportionally to the sizes of the larger
  classes.
* A fixed random seed (42) guarantees reproducibility.

The loader never mutates the original graph files and never loads the entire
training set into memory unless the user explicitly asks for the full set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

__all__ = ["DataLoader"]


class DataLoader:
    """Utility class for streaming graph data.

    Parameters
    ----------
    data_root:
        Absolute path to the ``graph_data_bundle/data`` directory.  If omitted, a
        default relative to the repository root is used.
    split_name:
        Name of the sub-directory that contains the UNSW graphs - ``graphs_unsw``
        for the primary dataset.
    seed:
        Seed for all random operations (sub-sampling).  The default follows the
        project specification.
    """

    def __init__(
        self,
        data_root: Optional[Path] = None,
        split_name: str = "graphs_unsw",
        seed: int = 42,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.data_root = Path(data_root) if data_root is not None else repo_root / "graph_data_bundle" / "data"
        self.graph_dir: Path = self.data_root / split_name
        if not self.graph_dir.is_dir():
            raise FileNotFoundError(f"Graph directory '{self.graph_dir}' does not exist")
        self.seed = seed
        # Load class names once - needed for sanity checks and for the binary label mapping
        class_names_path = self.graph_dir / "class_names.json"
        if not class_names_path.is_file():
            raise FileNotFoundError(f"class_names.json not found in {self.graph_dir}")
        with class_names_path.open("r", encoding="utf-8") as f:
            self.class_names: List[str] = json.load(f)
        # Determine benign class index - used for binary attack metrics
        try:
            self.benign_id: int = self.class_names.index("Benign")
        except ValueError as exc:
            raise ValueError("'Benign' class not found in class_names.json") from exc

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _list_graph_files(self, split: str) -> List[Path]:
        """Return a deterministically sorted list of ``*.npz`` files for a split."""
        pattern = f"{split}_graph_*.npz"
        files = sorted(self.graph_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No graph files found for split '{split}' in {self.graph_dir}")
        return files

    def _load_npz(self, path: Path) -> dict:
        """Load a graph ``.npz`` file and return the underlying dict.

        The caller is responsible for closing the file - ``np.load`` returns a
        ``numpy.lib.npyio.NpzFile`` that behaves like a dict.
        """
        return np.load(path, allow_pickle=False)

    def _validate_graph(self, graph: dict, path: Path) -> None:
        """Validate shapes, feature count and binary-label consistency.

        Raises ``ValueError`` with a clear message if any check fails.
        """
        required_keys = {"edge_index", "edge_attr", "edge_label", "edge_attack", "num_nodes"}
        missing = required_keys - set(graph.keys())
        if missing:
            raise ValueError(f"Graph file {path} is missing keys: {missing}")

        edge_attr = graph["edge_attr"]
        if edge_attr.ndim != 2 or edge_attr.shape[1] != 39:
            raise ValueError(
                f"{path}: edge_attr must have shape (N, 39); got {edge_attr.shape}"
            )

        edge_attack = graph["edge_attack"]
        edge_label = graph["edge_label"]
        if edge_attack.shape != edge_label.shape:
            raise ValueError(
                f"{path}: edge_attack and edge_label must have identical shapes; got {edge_attack.shape} vs {edge_label.shape}"
            )

        # Verify binary consistency
        derived_binary = (edge_attack != self.benign_id).astype(np.int64)
        if not np.array_equal(derived_binary, edge_label):
            raise ValueError(
                f"{path}: edge_label does not match derived binary label from edge_attack"
            )

    # ---------------------------------------------------------------------
    # Public API - training data
    # ---------------------------------------------------------------------
    def _count_training_classes(self) -> Dict[int, int]:
        """Return a dict mapping class_id -> total edge count across the training split."""
        class_counts: Dict[int, int] = {}
        for npz_path in self._list_graph_files("train"):
            graph = self._load_npz(npz_path)
            self._validate_graph(graph, npz_path)
            attacks = graph["edge_attack"]
            for cid, cnt in zip(*np.unique(attacks, return_counts=True)):
                class_counts[int(cid)] = class_counts.get(int(cid), 0) + int(cnt)
        return class_counts

    def _determine_sampling_targets(
        self, class_counts: Dict[int, int], max_samples: Optional[int]
    ) -> Dict[int, int]:
        """Compute per-class sample targets respecting the max_samples budget.

        The policy follows the specification in the project description.
        """
        if max_samples is None or max_samples <= 0:
            # Use the full dataset
            return {cid: cnt for cid, cnt in class_counts.items()}

        total_available = sum(class_counts.values())
        if max_samples >= total_available:
            return {cid: cnt for cid, cnt in class_counts.items()}

        # Start with all classes; we will reduce to fit the budget.
        targets: Dict[int, int] = {cid: 0 for cid in class_counts}

        # First allocate full counts for small classes (<= 500) if possible
        remaining_budget = max_samples
        small_classes = [cid for cid, cnt in class_counts.items() if cnt <= 500]
        for cid in small_classes:
            alloc = min(class_counts[cid], remaining_budget)
            targets[cid] = alloc
            remaining_budget -= alloc

        # Allocate the rest proportionally among the remaining classes
        large_classes = [cid for cid in class_counts if cid not in small_classes]
        large_total = sum(class_counts[cid] for cid in large_classes)
        for cid in large_classes:
            proportion = class_counts[cid] / large_total if large_total > 0 else 0.0
            alloc = int(round(proportion * remaining_budget))
            # Do not allocate more than what is available for the class
            alloc = min(alloc, class_counts[cid])
            targets[cid] = alloc

        # Adjust for rounding errors - ensure the sum equals max_samples (or as close as possible)
        current_total = sum(targets.values())
        while current_total < max_samples:
            # Find a class that still has capacity
            capacities = {
                cid: class_counts[cid] - targets[cid] for cid in class_counts if class_counts[cid] > targets[cid]
            }
            if not capacities:
                break
            cid = max(capacities, key=capacities.get)
            targets[cid] += 1
            current_total += 1
        while current_total > max_samples:
            # Reduce from the class with the largest allocated size
            cid = max(targets, key=targets.get)
            if targets[cid] == 0:
                break
            targets[cid] -= 1
            current_total -= 1
        return targets

    def load_training(
        self,
        max_samples: int = 150_000,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """Load (optionally subsampled) training data.

        Returns ``X`` (shape ``(n_samples, 39)``), ``y`` (multiclass integer IDs)
        and a metadata dictionary describing the sampling policy and class
        distributions.
        """
        # First pass - count class frequencies
        class_counts = self._count_training_classes()
        # Determine per-class targets based on the requested budget
        sample_targets = self._determine_sampling_targets(class_counts, max_samples)

        rng = np.random.RandomState(self.seed)
        collected: Dict[int, int] = {cid: 0 for cid in class_counts}
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        # Second pass - stream graphs and collect samples per class
        for npz_path in self._list_graph_files("train"):
            graph = self._load_npz(npz_path)
            self._validate_graph(graph, npz_path)
            edge_attr = graph["edge_attr"]  # (N, 39)
            edge_attack = graph["edge_attack"]  # (N,)

            # Process each class present in this graph
            for cid in np.unique(edge_attack):
                cid_int = int(cid)
                needed = sample_targets.get(cid_int, 0) - collected.get(cid_int, 0)
                if needed <= 0:
                    continue
                idx = np.where(edge_attack == cid_int)[0]
                if len(idx) <= needed:
                    chosen = idx
                else:
                    chosen = rng.choice(idx, size=needed, replace=False)
                X_parts.append(edge_attr[chosen])
                y_parts.append(np.full(chosen.shape, cid_int, dtype=np.int64))
                collected[cid_int] = collected.get(cid_int, 0) + len(chosen)

            # Early exit if we already satisfied all targets
            if all(collected[c] >= sample_targets[c] for c in sample_targets):
                break

        if not X_parts:
            raise RuntimeError("No training samples were collected - check the sampling parameters.")

        X_train = np.vstack(X_parts).astype(np.float32)
        y_train = np.concatenate(y_parts).astype(np.int64)

        meta = {
            "original_class_counts": class_counts,
            "sampled_class_counts": {cid: collected.get(cid, 0) for cid in class_counts},
            "total_train_edges": sum(class_counts.values()),
            "samples_used": int(X_train.shape[0]),
            "max_samples_requested": max_samples,
            "sampling_policy": "retain_all_<=500_then_proportional",
            "benign_id": self.benign_id,
        }
        return X_train, y_train, meta

    # ---------------------------------------------------------------------
    # Public API - validation / test splits (full, no sampling)
    # ---------------------------------------------------------------------
    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the full feature matrix and labels for a given split.

        Parameters
        ----------
        split:
            One of ``"val"`` or ``"test"`` (``"train"`` is handled by
            ``load_training``).
        Returns
        -------
        X, y_multiclass, y_binary
            ``X`` - ``(N, 39)`` float32 matrix.
            ``y_multiclass`` - integer IDs.
            ``y_binary`` - ``0`` for benign, ``1`` for any attack (derived from
            ``y_multiclass`` using the stored ``benign_id``).
        """
        if split not in {"val", "test"}:
            raise ValueError("split must be 'val' or 'test'")
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for npz_path in self._list_graph_files(split):
            graph = self._load_npz(npz_path)
            self._validate_graph(graph, npz_path)
            X_parts.append(graph["edge_attr"])
            y_parts.append(graph["edge_attack"])
        if not X_parts:
            raise RuntimeError(f"No {split} graphs found in {self.graph_dir}")
        X = np.vstack(X_parts).astype(np.float32)
        y_multi = np.concatenate(y_parts).astype(np.int64)
        y_binary = (y_multi != self.benign_id).astype(np.int64)
        return X, y_multi, y_binary

    # ---------------------------------------------------------------------
    # Extended API - split loading with graph/edge identifiers
    # ---------------------------------------------------------------------
    def load_split_with_ids(
        self,
        split: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a split and also return per-row graph and edge identifiers.

        Returns ``X, y_multi, y_binary, graph_ids, edge_ids`` where ``graph_ids``
        contains the integer index of the source ``*.npz`` file (starting at ``0``)
        and ``edge_ids`` is the row offset within that graph.  This method is
        intended for exporting validation predictions while preserving the exact
        ordering required by the downstream evaluation scripts.
        """
        if split not in {"val", "test"}:
            raise ValueError("split must be 'val' or 'test'")
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        graph_id_parts: List[np.ndarray] = []
        edge_id_parts: List[np.ndarray] = []
        for graph_idx, npz_path in enumerate(self._list_graph_files(split)):
            graph = self._load_npz(npz_path)
            self._validate_graph(graph, npz_path)
            X = graph["edge_attr"]
            y_multi = graph["edge_attack"]
            n = X.shape[0]
            X_parts.append(X)
            y_parts.append(y_multi)
            graph_id_parts.append(np.full(n, graph_idx, dtype=np.int64))
            edge_id_parts.append(np.arange(n, dtype=np.int64))
        if not X_parts:
            raise RuntimeError(f"No {split} graphs found in {self.graph_dir}")
        X = np.vstack(X_parts).astype(np.float32)
        y_multi = np.concatenate(y_parts).astype(np.int64)
        graph_ids = np.concatenate(graph_id_parts).astype(np.int64)
        edge_ids = np.concatenate(edge_id_parts).astype(np.int64)
        y_binary = (y_multi != self.benign_id).astype(np.int64)
        return X, y_multi, y_binary, graph_ids, edge_ids
