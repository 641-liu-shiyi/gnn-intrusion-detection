#!/usr/bin/env bash
# Checkpoint II, Member 3: everything after the ablations, in order. Safe to rerun: finished runs are skipped.
# 1. class weighting x classifier input (UNSW, 3 seeds), ranked on validation macro F1 (src/eval/metrics)
# 2. final model = best of (1); CIC-IDS2018 model with the same settings (1 seed)
# 3. cross-dataset transfer both ways, val and test
# 4. error analysis and prediction exports (shared artefact format) for the final models
set -e
source ~/dlenv/bin/activate
cd ~/gnn-intrusion-detection

python member3/sweep.py member3/configs/sweep_unsw_weighting.json --run
python member3/summarize.py runs/sweep_unsw_weighting -val_at_best.macro_f1 > runs/weighting_ranked.txt

python - <<'PY'
import json, statistics as st
from collections import defaultdict
from pathlib import Path
scores = defaultdict(list)
for r in Path("runs/sweep_unsw_weighting").glob("*/results.json"):
    scores[r.parent.name.rsplit("_seed", 1)[0]].append(json.loads(r.read_text())["val_at_best"]["macro_f1"])
best = max(scores, key=lambda k: st.mean(scores[k]))
print("val macro F1 by config:", {k: round(st.mean(v), 4) for k, v in sorted(scores.items())})
print("FINAL MODEL:", best)
unsw = json.loads(Path(f"member3/configs/sweep_unsw_weighting/{best}.json").read_text())
cic = json.loads(Path("member3/configs/cicids_egraphsage_mean.json").read_text())
cic["model"], cic["train"]["class_weighting"] = unsw["model"], unsw["train"]["class_weighting"]
Path("member3/configs/cicids_final.json").write_text(json.dumps(cic, indent=2) + "\n")
Path("runs/final_unsw_run.txt").write_text(f"runs/sweep_unsw_weighting/{best}_seed0\n")
PY

UNSW_RUN=$(cat runs/final_unsw_run.txt)
[ -e runs/cicids_final_seed0/results.json ] || python member3/train.py member3/configs/cicids_final.json 0

for s in val test; do
  [ -e "$UNSW_RUN/transfer_to_cicids_egraphsage_mean_$s.json" ] || \
    python member3/transfer.py "$UNSW_RUN" member3/configs/cicids_egraphsage_mean.json $s
  [ -e "runs/cicids_final_seed0/transfer_to_unsw_egraphsage_mean_$s.json" ] || \
    python member3/transfer.py runs/cicids_final_seed0 member3/configs/unsw_egraphsage_mean.json $s
done

python member3/analyze.py "$UNSW_RUN" data/graphs_unsw/class_names.json > runs/final_unsw_error_analysis.txt
python member3/analyze.py runs/cicids_final_seed0 data/graphs_cicids/class_names.json > runs/final_cicids_error_analysis.txt
for s in val test; do
  [ -e "member3/predictions/$(basename $UNSW_RUN)_$s" ] || python member3/export_predictions.py "$UNSW_RUN" $s
  [ -e "member3/predictions/cicids_final_seed0_$s" ] || python member3/export_predictions.py runs/cicids_final_seed0 $s
done
echo CHECKPOINT2 ALL DONE
