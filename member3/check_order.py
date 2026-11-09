"""Read-only: are the rows behind the graphs in time order, or shuffled?
If shuffled, every graph's attack fraction sits near the dataset average (binomial noise only,
~0.5 percentage points for 5000 flows at 12%). If time-ordered, attacks arrive in bursts:
many graphs ~0% attack, some near 100%.
Usage: python member3/check_order.py"""
from pathlib import Path

import numpy as np

for name in ["graphs_unsw", "graphs_cicids"]:
    files = sorted(Path("data", name).glob("train_graph_*.npz"))
    frac = np.array([np.load(f)["edge_label"].mean() for f in files])
    p = frac.mean()
    print(f"\n{name}: {len(files)} train graphs, mean attack fraction {p:.2%}")
    print(f"  std across graphs: {frac.std():.2%}  (pure shuffle would give ~{np.sqrt(p * (1 - p) / 5000):.2%})")
    print(f"  graphs with  0% attack: {(frac == 0).mean():.0%}")
    print(f"  graphs with >50% attack: {(frac > 0.5).mean():.0%}")
    print(f"  percentiles 5/25/50/75/95: {np.round(np.percentile(frac, [5, 25, 50, 75, 95]) * 100, 1)}")
    print(f"  first 20 graphs (%): {np.round(frac[:20] * 100, 1)}")
