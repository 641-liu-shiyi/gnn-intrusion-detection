"""Error analysis for one finished run: where each class's test edges end up.
Usage: python member3/analyze.py <run_dir> [class_names.json]
class_names.json: list of names indexed by label id (from Member 1's label mapping).
Reported F1 comes only from Member 2's module (config "eval"); this script adds the confusion
breakdown and train-set frequency to explain *why* a class fails."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from train import load_evaluator  # noqa: E402

run = Path(sys.argv[1])
cfg = json.loads((run / "config.json").read_text())
res = json.loads((run / "results.json").read_text())
pred = np.load(run / "test_predictions.npz")
yt, yp = pred["y_true"], pred["y_pred"]
C = cfg["num_classes"]
names = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else [str(i) for i in range(C)]
train_counts = res["train_class_counts"]

evaluate = load_evaluator(cfg.get("eval", {}))
if evaluate:
    print("Member 2 metrics (test):", json.dumps(evaluate(yt, yp), default=float, indent=2))

cm = confusion_matrix(yt, yp, labels=list(range(C)))
np.savetxt(run / "confusion_test.csv", cm, fmt="%d", delimiter=",",
           header=",".join(names), comments="")

print(f"\n{'class':20s} {'train n':>10s} {'test n':>9s}  test edges predicted as (top 3)")
for i in np.argsort(train_counts):  # rarest training class first
    n = cm[i].sum()
    if n == 0:
        print(f"{names[i]:20s} {train_counts[i]:>10,d} {0:>9d}  (absent from test)")
        continue
    top = np.argsort(cm[i])[::-1][:3]
    dest = ", ".join(f"{names[j]} {cm[i, j] / n:.0%}" for j in top if cm[i, j])
    flag = "  <- never seen in training" if train_counts[i] == 0 else ""
    print(f"{names[i]:20s} {train_counts[i]:>10,d} {n:>9,d}  {dest}{flag}")
print(f"\nconfusion matrix saved to {run / 'confusion_test.csv'} (rows = true, cols = predicted)")
