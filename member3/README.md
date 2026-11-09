# Member 3 — E-GraphSAGE edge classification

Graph-based intrusion detection on NetFlow **v2** (NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2).
Hosts are nodes, flows are directed edges, each flow is classified by attack type.
Inputs are Member 1's graph files; metrics come from Member 2's shared evaluation module.

## Files

| File | Purpose |
|---|---|
| `model.py` | E-GraphSAGE (Lo et al., 2022) with mean / sum / attention aggregation and ablation switches |
| `train.py` | Config-driven training: seeding, class-weighted loss, per-epoch log, early stopping, single test evaluation |
| `sweep.py` | Expands a sweep spec (`grid` or one-change `variants`) into config files and runs them; resumable |
| `summarize.py` | Mean ± std over seeds per config, optionally ranked by a validation field |
| `analyze.py` | Per-class error breakdown (where each class's test flows go) and confusion matrix |
| `phase1_verify_inputs.py` | Read-only check of Member 1's outputs |
| `smoke_test.py` | Throwaway synthetic-data test of the whole pipeline; deletes its data |
| `configs/` | Every config used for a reported number |

## Inputs expected

From Member 1, under the repo root:

- `data/graphs_unsw/{train,val,test}_graph_NNNNN.npz` with keys `edge_index`, `edge_attr`, `edge_label` (multi-class ids), `num_nodes`
- `data/features/feature_names.joblib` (list of feature names; training refuses to start on IP addresses, timestamps (`FLOW_START`/`FLOW_END`) or `MIN_TTL`/`MAX_TTL`)

From Member 2: a module and function set in each config's `"eval"` block, called as `evaluate(y_true, y_pred)` on numpy arrays and returning a dict.

## Model

- Node features are a constant 1 (no IP identity). All information comes from flow features.
- Layer: `h_v' = W_apply [h_v || AGG_u W_msg [h_u || e_uv]]`.
- Messages flow in both directions with a 0/1 direction flag, so hosts that only send (e.g. scanners) still aggregate their flows.
- A flow is classified from `[h_src || h_dst]` (plus `e_uv` when `head_edge_attr` is on).
- Attention: GAT-style score `LeakyReLU(a^T [W_q h_v || m_uv])`, softmax over each host's incoming flows.

## Reproducing results

```bash
source ~/dlenv/bin/activate && cd <repo root>
python member3/smoke_test.py                                    # sanity check, ~30 s
python member3/train.py member3/configs/<config>.json [seed]    # one run -> runs/<config>_seed<seed>/
python member3/sweep.py member3/configs/<spec>.json --run       # a sweep or ablation set
python member3/summarize.py runs/<dir> [sort_key]               # mean ± std over seeds
python member3/summarize.py runs/<dir> --latex=test.macro_f1,params   # LaTeX table (booktabs)
python member3/analyze.py runs/<run_dir> class_names.json       # error analysis
```

Each run folder holds `config.json` (exact config incl. seed), `epochs.jsonl` (per-epoch log),
`best.pt` (checkpoint with lowest val loss), `results.json` and `test_predictions.npz`.

## Experiment protocol

- Hyperparameters and variants are chosen on **validation** fields only (`best_val_loss`, `val_at_best.*`); `test` is reported, never used for selection.
- Comparisons use 3 seeds and report mean ± std.
- Phase 4 is two stages: architecture × lr with fixed weighting (ranked by val loss, comparable since the loss is identical), then class weighting (ranked by val macro F1, since weighting changes the loss scale).
- Mean vs attention: report parameter counts (attention has ~22% more at hidden 128).
- GPU scatter is not bitwise deterministic; seeded results match to small numerical noise.

## Cross-dataset protocol (team decision: option C)

The two datasets share almost no attack types (only Benign and DoS), so cross-dataset results are
benign vs attack. The source model is trained multi-class as usual; on the target, any non-benign
prediction counts as "attack". The same collapse is applied to the source's own test set, so the
in-domain and cross-domain binary numbers come from one model.

```bash
python member3/transfer.py runs/<unsw_run> member3/configs/cicids_egraphsage_mean.json
python member3/transfer.py runs/<cicids_run> member3/configs/unsw_egraphsage_mean.json
```

Requires `data.benign_class` in both configs and identical feature lists (names and order).

## Status

Code complete and smoke-tested for phases 2–7 (phase 7: README and LaTeX export only).
Waiting on: Member 1's regenerated v2 graphs (multi-class labels, TTL removed, class mapping),
Member 2's evaluation module, and Member 1's answer on which scaler the target dataset uses.
