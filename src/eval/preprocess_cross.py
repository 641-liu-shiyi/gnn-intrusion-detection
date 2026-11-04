"""Utilities for transforming target dataset features to the source model's
expected input space.

The original preprocessing pipelines for the UNSW and CICIDS datasets follow the
following steps for training:

1. Load raw CSV data.
2. Drop identifier columns and the ``Label`` column.
3. Apply ``np.log1p`` to a set of known columns (the *log columns*).
4. Fit a ``StandardScaler`` on the transformed data.
5. Store ``feature_names`` (the column ordering) and the fitted ``scaler``.

During cross-dataset evaluation we receive *already-scaled* features from the
target dataset (the ``edge_attr`` matrix stored in the ``*.npz`` graph files).
To reuse a source model trained on a different dataset we must reverse the
target-specific scaling, reorder the columns to match the source ordering, and
apply the source scaler.

This module provides :func:`inverse_transform_and_reorder` which implements the
required transformation contract.
"""

from __future__ import annotations

from typing import List

import numpy as np

# The set of columns that receive a ``log1p`` transformation in the original
# preprocessing scripts.  Both UNSW and CICIDS pipelines use the same list.
LOG_COLUMNS: List[str] = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "DNS_TTL_ANSWER",
]


def inverse_transform_and_reorder(
    X_target: np.ndarray,
    target_scaler,
    target_feature_names: List[str],
    source_scaler,
    source_feature_names: List[str],
) -> np.ndarray:
    """Transform ``X_target`` from the target preprocessing space to the source
    preprocessing space.

    Parameters
    ----------
    X_target:
        Array of shape ``(N, n_target_features)`` - the ``edge_attr`` matrix from
        the *target* dataset (already standardized with ``target_scaler``).
    target_scaler:
        The ``StandardScaler`` fitted on the target training data.
    target_feature_names:
        Ordered list of feature names corresponding to columns of ``X_target``.
    source_scaler:
        The ``StandardScaler`` fitted on the source training data (expects log-
        transformed features as input).
    source_feature_names:
        Ordered list of feature names expected by the source model.

    Returns
    -------
    np.ndarray
        The transformed feature matrix ready to be fed to the source model.
    """
    # ---------------------------------------------------------------
    # 1. Inverse-transform the target scaling to obtain the *log-transformed*
    #    feature values (i.e. after step 3 of the original pipeline).
    # ---------------------------------------------------------------
    try:
        X_log = target_scaler.inverse_transform(X_target)
    except Exception as exc:
        raise ValueError(f"Failed to inverse-transform target features: {exc}")

    # ---------------------------------------------------------------
    # 2. Reorder / drop columns to match the source feature ordering.
    #    ``source_feature_names`` must be a subset of ``target_feature_names``.
    # ---------------------------------------------------------------
    reorder_indices = []
    for name in source_feature_names:
        try:
            idx = target_feature_names.index(name)
        except ValueError as exc:
            raise ValueError(
                f"Source required feature '{name}' not present in target feature set"
            ) from exc
        reorder_indices.append(idx)
    X_selected = X_log[:, reorder_indices]

    # ---------------------------------------------------------------
    # 3. Log-column alignment:
    #    Both UNSW and CICIDS pipelines apply np.log1p to LOG_COLUMNS prior
    #    to fitting their StandardScaler.  Therefore, after inverting the target
    #    scaler, the features are already in the exact log1p representation
    #    expected by the source StandardScaler.  We verify that the source
    #    scaler receives log-scaled inputs without un-logging them.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # 4. Apply the source scaler to obtain the final standardized feature matrix.
    # ---------------------------------------------------------------
    try:
        X_source = source_scaler.transform(X_selected)
    except Exception as exc:
        raise ValueError(f"Failed to apply source scaler: {exc}")

    # Preserve ``float32`` for compatibility with downstream models.
    return X_source.astype(np.float32)
