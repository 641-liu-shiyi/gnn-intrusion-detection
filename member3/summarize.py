"""Mean ± std over seeds for every config in a runs folder.
Usage: python member3/summarize.py <runs_dir> [sort_key] [--latex=key1,key2,...]
sort_key ranks configs by that field's mean, e.g. best_val_loss (ascending)
or -val_at_best.macro_f1 (leading "-" = descending). Rank on val_* only, never test.
--latex prints a booktabs table (needs \\usepackage{booktabs}) with the chosen fields as columns."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)):
            yield f"{prefix}{k}", v


latex = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--latex=")), None)
args = [a for a in sys.argv[1:] if not a.startswith("--latex=")]

groups = defaultdict(list)
for f in sorted(Path(args[0]).glob("*/results.json")):
    groups[re.sub(r"_seed\d+$", "", f.parent.name)].append(json.loads(f.read_text()))

table = {}
for name, runs in groups.items():
    vals = defaultdict(list)
    for r in runs:
        for k, v in flat(r):
            vals[k].append(v)
    table[name] = vals

order = sorted(table)
if len(args) > 1:
    key = args[1].lstrip("-")
    order.sort(key=lambda n: np.mean(table[n][key]), reverse=args[1].startswith("-"))

if latex:
    esc = lambda s: s.replace("_", r"\_")  # noqa: E731
    print(r"\begin{tabular}{l" + "c" * len(latex) + "}\n\\toprule")
    print("Configuration & " + " & ".join(esc(k) for k in latex) + r" \\" + "\n\\midrule")
    for name in order:
        cells = [f"${np.mean(table[name][k]):.3f} \\pm {np.std(table[name][k]):.3f}$"
                 if table[name].get(k) else "--" for k in latex]
        print(esc(name) + " & " + " & ".join(cells) + r" \\")
    print("\\bottomrule\n\\end{tabular}")
    sys.exit()

for rank, name in enumerate(order, 1):
    vals = table[name]
    print(f"\n#{rank} {name}  (n={len(vals['test_loss'])} seeds)")
    for k, v in vals.items():
        print(f"  {k:32s} {np.mean(v):.4f} ± {np.std(v):.4f}")
