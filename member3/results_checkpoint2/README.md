# Member 3 (E-GraphSAGE) — Checkpoint II results

All numbers use the team's graph data (NF-UNSW-NB15-v2, NF-CICIDS2018-v3, 39 features, 5,000-flow windows)
and are computed with `src/eval/metrics.compute_metrics`. GNN rows are mean ± std over 3 seeds unless noted.
Model selection used validation data only.

## Final model

E-GraphSAGE, 2 layers, 128 hidden units, mean aggregation, bidirectional messages, classifier on
`[h_u, h_v, x_uv]` (flow features appended, Eq. 2), class weights = inverse square root of class frequency.
Config: `member3/configs/unsw_final.json` (reproduce: `python member3/train.py member3/configs/unsw_final.json <seed>`). Predictions: `member3/predictions/head_edge_attrTrue_winverse_sqrt_seed0_{val,test}`.

## GNN vs baselines (NF-UNSW-NB15-v2, test)

| Model | Macro F1 | Weighted F1 | Accuracy | Binary F1 |
|---|---|---|---|---|
| E-GraphSAGE as published (classifier on `[h_u, h_v]`) | 0.282 ± 0.008 | 0.940 | 0.931 | 0.874 |
| **E-GraphSAGE final** (flow features in classifier, sqrt weighting) | **0.511 ± 0.003** | **0.971** | **0.969** | **0.953** |
| Random Forest (Member 2) | 0.605 | 0.979 | 0.976 | 0.948 |
| XGBoost (Member 2) | 0.652 | 0.982 | 0.978 | 0.952 |
| MLP (Member 2) | 0.508 | 0.973 | 0.967 | 0.910 |

The final GNN matches the best baseline on benign-vs-attack detection (binary F1 0.953 vs 0.952) but trails the
tree ensembles on attack-type classification (macro F1 0.511 vs 0.652).

## Class weighting × classifier input (UNSW, 3 seeds)

| Classifier input | Weighting | Val macro F1 | Test macro F1 | Test binary F1 |
|---|---|---|---|---|
| `[h_u, h_v, x_uv]` | inverse sqrt | **0.506** | **0.511 ± 0.003** | 0.953 |
| `[h_u, h_v, x_uv]` | none | 0.463 | 0.467 ± 0.014 | 0.958 |
| `[h_u, h_v, x_uv]` | inverse | 0.425 | 0.431 ± 0.009 | 0.885 |
| `[h_u, h_v]` | inverse sqrt | 0.327 | 0.337 ± 0.009 | 0.924 |
| `[h_u, h_v]` | none | 0.283 | 0.290 ± 0.001 | 0.952 |
| `[h_u, h_v]` | inverse | 0.279 | 0.282 ± 0.008 | 0.874 |

Full inverse-frequency weights make the rarest classes (Worms: 71 training flows) dominate the loss and
are over-predicted; the square-root scheme is best for both classifier inputs.

## Ablations (UNSW, inverse weighting, 3 seeds, test macro F1)

| Variant | Macro F1 |
|---|---|
| Flow features in classifier | 0.431 ± 0.009 |
| No graph (0 layers, flow features only) | 0.359 ± 0.007 |
| Attention aggregation | 0.293 ± 0.009 |
| One-directional messages | 0.290 ± 0.018 |
| One layer | 0.287 ± 0.007 |
| Baseline (mean, as published) | 0.282 ± 0.008 |
| Sum aggregation | 0.107 ± 0.009 (diverged; needs degree normalisation) |

- The published classifier gives every flow between the same host pair the same prediction; UNSW windows have
  ~30 hosts for 5,000 flows, so adding the flow features is the largest gain.
- Graph vs flow-level, same linear classifier: 0.431 vs 0.359 macro F1 (binary F1 0.885 vs 0.700).
- Attention vs mean: +0.011, within about one standard deviation — no significant gain on UNSW.
- Architecture sweep (27 settings: 1–3 layers, 64/128/256 hidden, lr 3e-3/1e-3/3e-4): test macro F1 0.267–0.297.

## Error analysis (final model, UNSW test)

| Class | Train flows | Test F1 | Test flows predicted as (top 3) |
|---|---|---|---|
| Benign | 1,625,729 | 0.996 | Benign 99% |
| Generic | 8,487 | 0.776 | Generic 67%, Backdoor 12%, Exploits 12% |
| Exploits | 14,143 | 0.753 | Exploits 74%, Reconnaissance 7%, Fuzzers 7% |
| Fuzzers | 10,719 | 0.704 | Fuzzers 79%, Backdoor 9%, Reconnaissance 6% |
| Reconnaissance | 5,591 | 0.660 | Reconnaissance 71%, Backdoor 15%, Fuzzers 9% |
| Shellcode | 646 | 0.611 | Shellcode 76%, Reconnaissance 15%, Fuzzers 4% |
| Worms | 71 | 0.211 | Worms 40%, Exploits 22%, Reconnaissance 15% |
| Backdoor | 1,058 | 0.168 | Backdoor 63%, Reconnaissance 9%, DoS 8% |
| Analysis | 952 | 0.150 | **Backdoor 72%**, Analysis 9%, Exploits 9% |
| DoS | 2,604 | 0.081 | **Exploits 38%, Backdoor 32%**, Reconnaissance 8% |

Failure cases: Analysis, Backdoor and DoS are mostly confused with each other and with Exploits (low precision
for Backdoor), and Worms has too few training flows. Full matrix: `final_unsw_confusion_test.csv`.

## NF-CICIDS2018-v3 (in-domain) — limitation

With the current chronological split, Bot and Infilteration have no training flows, and the validation and test
attacks are only these two classes. The GNN, like the baselines (macro F1 ≈ 0.066), cannot detect them:
macro F1 0.065, binary F1 0.001 (seed 0); early stopping selected epoch 1 because validation contains no
attack class seen in training. CIC-IDS2018 results need a split in which every attack type appears in training.

## Cross-dataset (binary benign vs attack; target re-scaled with the source scaler, as in `src/eval/cross_dataset.py`)

| Direction | Split | Accuracy | Precision | Recall | Binary F1 | AP |
|---|---|---|---|---|---|---|
| UNSW → CICIDS | val | 0.875 | 0.083 | 0.319 | 0.131 | 0.051 |
| UNSW → CICIDS | test | 0.877 | 0.065 | 0.136 | 0.088 | 0.045 |
| CICIDS → UNSW | val | 0.319 | 0.027 | 0.266 | 0.049 | 0.061 |
| CICIDS → UNSW | test | 0.392 | 0.030 | 0.231 | 0.053 | 0.069 |

In-domain binary F1 of the same UNSW model: 0.947 (test). Transfer collapses in both directions, as for the
baselines (binary F1 0.04–0.12). UNSW → CICIDS is somewhat higher for the GNN than for the baselines
(0.088–0.131 vs 0.053–0.059); CICIDS → UNSW is limited by the CICIDS model itself (see limitation above).
CICIDS prediction exports (>100 MB) are kept on the server, not in the repository.
