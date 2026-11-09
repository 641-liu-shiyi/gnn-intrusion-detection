"""Baselines package for CS5814 Part 2.

The package currently provides a data-loader that streams flow-level features
directly from the pre-computed graph ``.npz`` files and a training script for a
multiclass Random Forest model.  The implementation deliberately avoids any
modification of the existing Part 1 scripts or the bundled dataset.

Only the public objects that are required by the CLI entry point are exported
here.  Sub-modules are imported lazily by the scripts themselves.
"""

# Exported symbols (optional, for ``from src.baselines import *`` convenience)
__all__ = []
