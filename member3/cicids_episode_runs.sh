#!/usr/bin/env bash
# Final report, Member 3: CIC-IDS2018 reruns on Shiyi's episode-aware split (commit e3c6ff5).
# Old-split runs (runs/cicids_final_seed0 etc.) are left untouched. Safe to rerun: finished steps are skipped.
# 0. check the new data: every class in every split, Benign = 0, scaler + same 39 features as UNSW
# 1. CICIDS final settings (mean) and attention, 3 seeds each
# 2. transfer both ways, val and test, 3 seeds (UNSW final seeds -> CICIDS; CICIDS final seeds -> UNSW)
# 3. summaries, error analysis, prediction exports (seed 0)
set -e
source ~/dlenv/bin/activate
cd ~/gnn-intrusion-detection

OUT=runs/cicids_episode
CIC_FINAL=member3/configs/cicids_episode_final.json
CIC_ATT=member3/configs/cicids_episode_attention.json
UNSW_TGT=member3/configs/unsw_final.json
UNSW_BASE=runs/sweep_unsw_weighting/head_edge_attrTrue_winverse_sqrt   # + _seed{0,1,2}
SEEDS="0 1 2"

python - <<'PY'
import json, sys
from pathlib import Path
import joblib, numpy as np
g, f = Path("data/graphs_cicids_episode"), Path("data/features_cicids_episode")
for p in (g / "class_names.json", f / "scaler.joblib", f / "feature_names.joblib"):
    if not p.exists():
        sys.exit(f"STOP: {p} missing - new CICIDS data not in place yet.")
names = json.loads((g / "class_names.json").read_text())
if len(names) != 15 or names[0] != "Benign":
    sys.exit(f"STOP: expected 15 classes with Benign = 0, got {names}")
if list(joblib.load(f / "feature_names.joblib")) != list(joblib.load("data/features/feature_names.joblib")):
    sys.exit("STOP: CICIDS feature names differ from UNSW's.")
bad = False
for split in ("train", "val", "test"):
    files = sorted(g.glob(f"{split}_graph_*.npz"))
    counts = np.zeros(len(names), dtype=np.int64)
    for p in files:
        with np.load(p) as d:
            counts += np.bincount(d["edge_attack"], minlength=len(names))
    missing = [n for n, c in zip(names, counts) if c == 0]
    print(f"{split}: {len(files)} graphs | " + ", ".join(f"{n} {c:,}" for n, c in zip(names, counts)))
    if missing:
        print(f"  MISSING in {split}: {missing}")
        bad = True
if bad:
    sys.exit("STOP: some classes have no flows in a split; tell Shiyi before training.")
print("data check OK")
PY

# 1. training
for s in $SEEDS; do
  for cfg in $CIC_FINAL $CIC_ATT; do
    run="$OUT/$(basename "$cfg" .json)_seed$s"
    [ -e "$run/results.json" ] || python member3/train.py "$cfg" "$s"
  done
done

# 2. transfer (UNSW final runs must already exist from Checkpoint II)
for s in $SEEDS; do
  CIC_RUN="$OUT/cicids_episode_final_seed$s"
  for split in val test; do
    if [ -d "${UNSW_BASE}_seed$s" ]; then
      [ -e "${UNSW_BASE}_seed$s/transfer_to_cicids_episode_final_$split.json" ] || \
        python member3/transfer.py "${UNSW_BASE}_seed$s" $CIC_FINAL $split
    else
      echo "WARNING: ${UNSW_BASE}_seed$s not found; skipping UNSW->CICIDS for seed $s"
    fi
    [ -e "$CIC_RUN/transfer_to_unsw_final_$split.json" ] || \
      python member3/transfer.py "$CIC_RUN" $UNSW_TGT $split
  done
done

# 3. summaries, error analysis, exports
python member3/summarize.py $OUT -val_at_best.macro_f1 > $OUT/summary.txt
python member3/analyze.py $OUT/cicids_episode_final_seed0 data/graphs_cicids_episode/class_names.json \
  > $OUT/error_analysis_final_seed0.txt
for split in val test; do
  [ -e "member3/predictions/cicids_episode_final_seed0_$split" ] || \
    python member3/export_predictions.py $OUT/cicids_episode_final_seed0 $split
done
cat $OUT/summary.txt
echo CICIDS EPISODE ALL DONE
