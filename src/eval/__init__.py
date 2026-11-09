"""Evaluation utilities for CS5814 Part 2.

The ``metrics`` module supplies a single public function ``compute_metrics``
that works with any multiclass predictions (Random Forest, XGBoost, MLP, GNN,
etc.).  It does **not** depend on the training code and can be reused across
different baseline implementations.
"""

__all__ = ["metrics"]
