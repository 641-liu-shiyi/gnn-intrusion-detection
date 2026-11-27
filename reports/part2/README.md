# CS5814 Deep Learning — Part 2 Publication-Ready Research Artifacts

**Project Title:** Graph-Based Network Intrusion Detection  
**Author / Responsibility:** Part 2 Lead Software Engineer  
**Institution:** Cornell University / CS 5814 Deep Learning  
**Git Branch:** `feature/cicids-episode-baselines`  
**Package Scope:** Lightweight Git-friendly research deliverables containing primary evaluation tables, IEEE-compatible LaTeX tables, class-level breakdowns, training budgets, model timing, and diagnostic plots.

---

## 1. Dataset & Sample Count Overview

All baseline models evaluate standardized 39-feature NetFlow v2 representations:

1. **NF-UNSW-NB15-v2** (`graph_data_bundle/data/graphs_unsw/`, preserved):
   - **Training Set**: 334 graphs (**1,670,000 flows**)
   - **Validation Set**: 47 graphs (**235,000 flows**; 15,511 attack, 219,489 benign)
   - **Held-Out Test Set**: 95 graphs (**475,000 flows**; 34,835 attack, 440,165 benign)
   - **Class Space (10 classes)**: *Analysis, Backdoor, Benign, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms*
   - **Benign Class Mapping**: Index `2` = `"Benign"` (Attack = $y \neq 2$)

2. **NF-CSE-CIC-IDS2018-v2 — Corrected Episode-Aware Split** (`data/graphs_cicids_episode/` & `data/features_cicids_episode/`):
   - **Training Set (Sampled Budget)**: **100,000 flows** (CPU compute-bounded budget; `0.752%` of 13,295,000 training flows across 2,659 graphs)
   - **Validation Set**: 342 graphs (**1,710,000 flows**; 209,425 attack, 1,500,575 benign).
   - **Held-Out Test Set**: 895 graphs (**4,475,000 flows**; 418,823 attack, 4,056,177 benign).
   - **Class Space (15 classes)**: *Benign, Bot, Brute_Force_-Web, Brute_Force_-XSS, DDOS_attack-HOIC, DDOS_attack-LOIC-UDP, DDoS_attacks-LOIC-HTTP, DoS_attacks-GoldenEye, DoS_attacks-Hulk, DoS_attacks-SlowHTTPTest, DoS_attacks-Slowloris, FTP-BruteForce, Infilteration, SQL_Injection, SSH-Bruteforce*
   - **Training Class Presence**: **All 15 classes** are present in `train`, `val`, and `test` (superseding the obsolete chronological split where `Bot` and `Infilteration` had 0 training flows).
   - **Benign Class Mapping**: Index `0` = `"Benign"` (Attack = $y \neq 0$)

*Note*: Obsolete chronological-split CICIDS reports have been archived under `reports/part2/archive_chronological_split/`.

---

## 2. Model Architectures & Hyperparameters

| Hyperparameter / Setting | Random Forest | XGBoost | PyTorch MLP |
|---|---|---|---|
| **Estimators / Epochs** | 50 trees | 50 boosting rounds | 10 epochs (early stopping patience 3) |
| **Max Depth / Hidden Dims** | `max_depth=15` | `max_depth=6` | Two hidden layers: `64, 32` |
| **Learning Rate** | N/A | $\eta = 0.1$ | $\alpha = 0.001$ (Adam optimizer) |
| **Subsample / Batch Size** | Bootstrap sample | `tree_method="hist"` | `batch_size=512` |
| **Regularization** | `min_samples_leaf=2` | Histogram binning (`hist`) | Dropout ($p=0.2$) |
| **Class Weighting Policy** | `balanced_subsample` | `balanced` sample weights | Aligned full-space balanced weights in CrossEntropyLoss |
| **Primary Random Seed** | `seed=42` | `seed=42` | `seed=42` |
| **Multi-Seed Stability** | Seeds `42, 43, 44` (UNSW 50k & CICIDS 100k) | Single seed (`42`) | Single seed (`42`) |

---

## 3. Directory Layout & Deliverables

- `CICIDS_EPISODE_DATA_AUDIT.md`: Corrected CICIDS episode dataset & scaler audit
- `CICIDS_EPISODE_RESULTS.md`: Summary of CICIDS episode baselines, ablations, transfer, and GNN comparison
- `DATA_COVERAGE_AUDIT.md`: Physical dataset and coverage audit
- `tables/`:
  - `final_test_summary.csv`: UNSW (475k) and CICIDS Episode (4.475M) held-out test summary
  - `baseline_comparison_unsw.{csv,md,tex}` & `val_vs_test_unsw.{csv,md,tex}` (preserved UNSW)
  - `baseline_comparison_cicids.{csv,md,tex}` (CICIDS episode validation), `baseline_comparison_cicids_test.{csv,md,tex}` (CICIDS episode test), `val_vs_test_cicids.{csv,md,tex}`
  - `unsw_test_per_class.{csv,md}` & `cicids_test_per_class.{csv,md,tex}`
  - `cross_dataset_comparison.{csv,md,tex}` (UNSW <-> CICIDS Episode on both `val` and `test`)
  - `ablation_class_weight_summary.csv` (UNSW) & `ablation_class_weight_cicids_summary.csv` (CICIDS Episode)
  - `multiseed_stability_rf_summary.csv` (UNSW) & `multiseed_stability_cicids_rf_summary.csv` (CICIDS Episode)
  - `gnn_vs_baselines_cicids_episode.{csv,md,tex}`
  - `training_sample_budget_summary.csv` & `model_timing_summary.csv`
- `per_class/`: Detailed per-class precision, recall, F1, and support for UNSW and CICIDS Episode test
- `figures/`: Confusion matrices, PR curves (UNSW + CICIDS Episode test), GNN comparison chart, and CICIDS class-weighting ablation chart
- `archive_chronological_split/`: Archived old chronological-split CICIDS tables, per-class CSVs, and figures

---

## 4. Reproducibility Commands

```powershell
# Audit physical datasets
uv run --locked python -m scripts.audit_physical_data

# Regenerate comparison tables and package reports
uv run --locked python -m scripts.generate_all_reports
uv run --locked python -m scripts.package_part2_reports

# Recompute all metrics directly from saved predictions.npz
uv run --locked python -m scripts.audit_full_experiments

# Run automated Part 2 tests
uv run --locked python -m pytest tests --ignore=tests/test_explain.py -q
```
