# CS5814 Deep Learning — Part 2 Publication-Ready Research Artifacts

**Project Title:** Graph-Based Network Intrusion Detection  
**Author / Responsibility:** Part 2 Lead Software Engineer  
**Institution:** Cornell University / CS 5814 Deep Learning  
**Git Branch:** `feature/baselines-evaluation`  
**Package Scope:** Lightweight (<250 KB) Git-friendly research deliverables containing primary evaluation tables, IEEE-compatible LaTeX tables, class-level breakdowns, training budgets, model timing, and key diagnostic plots.

---

## 1. Dataset & Sample Count Overview

All baseline models evaluate standardized 39-feature NetFlow v2 representations:

1. **NF-UNSW-NB15-v2**:
   - **Training Set**: 334 graphs (**1,670,000 flows**)
   - **Validation Set**: 47 graphs (**235,000 flows**; 15,511 attack, 219,489 benign)
   - **Held-Out Test Set**: 95 graphs (**475,000 flows**; 34,835 attack, 440,165 benign)
   - **Class Space (10 classes)**: *Analysis, Backdoor, Benign, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms*
   - **Benign Class Mapping**: Index `2` = `"Benign"` (Attack = $y \neq 2$)

2. **NF-CSE-CIC-IDS2018-v2**:
   - **Training Set (Sampled Budget)**: **100,000 flows** (CPU compute-bounded budget; 0.733% of 13,640,000 training flows across 2,728 graphs; 0.513% of the ~19.5M flow total dataset)
   - **Validation Set**: 389 graphs (**1,945,000 flows**; 57,343 attack, 1,887,657 benign). Majority-class baseline: **97.05%**.
   - **Held-Out Test Set**: 779 graphs (**3,895,000 flows**; 169,896 attack, 3,725,104 benign). Majority-class baseline: **95.64%**.
   - **Class Space (15 classes defined in taxonomy)**: *Benign, Bot, Brute Force -Web, Brute Force -XSS, DDOSe-LOIC-HTTP, DDoS attacks-LOIC-HTTP, DDoS-HOIC, DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris, FTP-BruteForce, Infilteration, SQL Injection, SSH-Bruteforce*
   - **Training Class Presence**: Exactly **13 classes** are physically present in the training partition (1 Benign + 12 Attack classes). 2 classes (`Bot` and `Infilteration`) have 0 flows in the raw training graphs due to chronological day-based splitting. All 13 available classes were preserved and sampled in the 100k budget.
   - **Benign Class Mapping**: Index `0` = `"Benign"` (Attack = $y \neq 0$)
   - **Class Imbalance & Majority Baselines**: 
     - Validation: 97.05% Benign, 2.95% Attack (Majority-class accuracy baseline: **97.05%**)
     - Held-out Test: 95.64% Benign, 4.36% Attack (Majority-class accuracy baseline: **95.64%**)

---

## 2. Model Architectures & Hyperparameters

All baseline models are deterministic and frozen following validation selection:

| Hyperparameter / Setting | Random Forest | XGBoost | PyTorch MLP |
|---|---|---|---|
| **Estimators / Epochs** | 50 trees | 50 boosting rounds | 10 epochs (early stopping patience 3) |
| **Max Depth / Hidden Dims** | `max_depth=15` | `max_depth=6` | Two hidden layers: `64, 32` |
| **Learning Rate** | N/A | $\eta = 0.1$ | $\alpha = 0.001$ (Adam optimizer) |
| **Subsample / Batch Size** | Bootstrap sample | `subsample=0.8, colsample=0.8` | `batch_size=512` |
| **Regularization** | `min_samples_split=10, min_samples_leaf=5` | Histogram binning (`hist`) | Dropout ($p=0.2$), BatchNorm, weight decay $10^{-4}$ |
| **Class Weighting Policy** | `balanced_subsample` | `balanced` sample weights ($w_i = \frac{N}{C \cdot N_{y_i}}$) | Aligned full-space balanced weights in CrossEntropyLoss |
| **Primary Random Seed** | `seed=42` | `seed=42` | `seed=42` |
| **Multi-Seed Stability** | Seeds `42, 43, 44` (50k budget) | Single seed (42) | Single seed (42) |

---

## 3. Directory Layout & Deliverables

```
reports/part2/
├── README.md                                  <- Dataset specifications, hyperparameters & reproduction guide
├── tables/                                    <- CSV, LaTeX, and Markdown summary tables
│   ├── final_test_summary.csv                 <- Full test split metrics across all models & datasets
│   ├── baseline_comparison_unsw.{csv,tex,md}  <- Full UNSW validation baseline comparison
│   ├── baseline_comparison_cicids.{csv,tex,md}<- CICIDS validation baseline comparison
│   ├── val_vs_test_unsw.{csv,tex,md}          <- Side-by-side UNSW validation vs held-out test tracking
│   ├── val_vs_test_cicids.{csv,tex,md}        <- Side-by-side CICIDS validation vs held-out test tracking
│   ├── unsw_test_per_class.{csv,md}           <- Class-by-class metrics across all 10 UNSW attack classes
│   ├── cicids_test_per_class.csv              <- Class-by-class metrics across all 15 CICIDS attack classes
│   ├── cross_dataset_comparison.{csv,tex,md}  <- Bidirectional domain transfer results (UNSW <-> CICIDS)
│   ├── ablation_class_weight_summary.csv      <- Single-factor class weighting ablation table
│   ├── multiseed_stability_rf_summary.csv     <- 3-seed Random Forest stability summary
│   ├── training_sample_budget_summary.csv     <- Sample-size scaling curve (2k, 20k, 50k, 100k, full)
│   └── model_timing_summary.csv               <- Training duration and inference throughput
├── per_class/                                 <- Detailed per-class precision, recall, F1, and support
│   ├── unsw_test_rf_per_class.csv
│   ├── unsw_test_xgb_per_class.csv
│   ├── unsw_test_mlp_per_class.csv
│   ├── cicids_test_rf_per_class.csv
│   ├── cicids_test_xgb_per_class.csv
│   └── cicids_test_mlp_per_class.csv
└── figures/                                   <- High-resolution diagnostic plots (<60 KB each)
    ├── unsw_test_rf_confusion_matrix_binary.png
    ├── unsw_test_xgb_confusion_matrix_binary.png
    ├── unsw_test_mlp_confusion_matrix_binary.png
    ├── unsw_test_xgb_confusion_matrix_normalized.png
    ├── cicids_test_xgb_confusion_matrix_binary.png
    ├── unsw_test_rf_pr_curve.png
    ├── unsw_test_xgb_pr_curve.png
    └── unsw_test_mlp_pr_curve.png
```

---

## 4. Experiment Scope & Mapping to Local Results Artifacts

All deliverables trace directly to persisted, Git-ignored binary checkpoints and full prediction arrays in `results/`:

| Experiment Scope | Model(s) | Dataset & Budget | Packaged Output in `reports/part2/` | Original Local Source in `results/` |
|---|---|---|---|---|
| **UNSW Full Validation** | RF, XGB, MLP | UNSW Full (1.67M train / 235k val) | `tables/baseline_comparison_unsw.*` | `results/experiments/unsw_full_seed42/{rf,xgb,mlp}/` |
| **UNSW Held-Out Test** | RF, XGB, MLP | UNSW Full Test (475,000 flows) | `tables/final_test_summary.csv` | `results/test_evaluation/unsw/{rf,xgb,mlp}/` |
| **CICIDS Validation** | RF, XGB, MLP | CICIDS 100k (100k train / 1.945M val)| `tables/baseline_comparison_cicids.*` | `results/experiments/cicids_100k_seed42/{rf,xgb,mlp}/` |
| **CICIDS Held-Out Test** | RF, XGB, MLP | CICIDS Full Test (3,895,000 flows) | `tables/final_test_summary.csv` | `results/test_evaluation/cicids/{rf,xgb,mlp}/` |
| **UNSW $\to$ CICIDS Transfer**| RF, XGB, MLP | Cross-Dataset (1,945,000 target flows)| `tables/cross_dataset_comparison.*` | `results/cross_dataset/unsw_to_cicids/{rf,xgb,mlp}/` |
| **CICIDS $\to$ UNSW Transfer**| RF, XGB, MLP | Cross-Dataset (235,000 target flows) | `tables/cross_dataset_comparison.*` | `results/cross_dataset/cicids_to_unsw/{rf,xgb,mlp}/` |
| **Class Weighting Ablation** | RF, XGB, MLP | UNSW 50k (`balanced` vs `none`) | `tables/ablation_class_weight_summary.csv` | `results/experiments/ablation_class_weight_{rf,xgb,mlp}/` |
| **Multi-Seed Stability** | Random Forest | UNSW 50k (Seeds `42, 43, 44`) | `tables/multiseed_stability_rf_summary.csv` | `results/experiments/multiseed_stability_rf/` |
| **Training Sample Budget** | RF, XGB, MLP | Subsample scaling (2k, 20k, 50k, 100k, full) | `tables/training_sample_budget_summary.csv` | Multiple local runs (`smoke_2k`, `smoke_20k`, `50k`, `full`) |
| **Model Timing & Speed** | RF, XGB, MLP | Train time & Inference flows/sec | `tables/model_timing_summary.csv` | Computed during training and evaluation runs |

---

## 5. Teammate Reproduction & Checkpoint Access

### A. Run Automated Functional Test Suite (Offline)
The test suite is maintained locally and runs without network access:
```powershell
uv run --locked python -m pytest tests -q
```
*Expected Result*: 32 passed, 0 failed in ~15-30 seconds.

### B. Evaluate Frozen Checkpoints on Held-Out Test Splits
```powershell
# Evaluate frozen UNSW models on test split
uv run --locked python -m src.eval.evaluate_checkpoint --model rf --dataset unsw --checkpoint results/experiments/unsw_full_seed42/rf/model.joblib --split test --confirm-test --output-dir results/test_evaluation/unsw/rf
uv run --locked python -m src.eval.evaluate_checkpoint --model xgb --dataset unsw --checkpoint results/experiments/unsw_full_seed42/xgb/model.joblib --split test --confirm-test --output-dir results/test_evaluation/unsw/xgb
uv run --locked python -m src.eval.evaluate_checkpoint --model mlp --dataset unsw --checkpoint results/experiments/unsw_full_seed42/mlp/model.pt --split test --confirm-test --output-dir results/test_evaluation/unsw/mlp
```

### C. Regenerate Summary Tables
```powershell
uv run --locked python -m scripts.generate_all_reports
uv run --locked python -m scripts.package_part2_reports
```

### D. Ingest External Teammate GNN (E-GraphSAGE) Predictions
When the teammate generates their E-GraphSAGE output directory with `metrics.json`, `run_config.json`, and `predictions.npz`:
```python
from pathlib import Path
from src.comparison.gnn_importer import load_gnn_artifacts
from src.comparison.report import compare_aggregate, compare_strict, load_results

gnn = load_gnn_artifacts(Path("path/to/teammate_gnn_output"))
rf = load_results(Path("results/experiments/unsw_full_seed42/rf"))
xgb = load_results(Path("results/experiments/unsw_full_seed42/xgb"))

print(compare_aggregate({"Random Forest": rf, "XGBoost": xgb, "E-GraphSAGE": gnn["metrics"]}))
```

### E. Obtaining Large Checkpoints & Prediction Arrays for Paired Testing
Due to GitHub file size limits, large binary model checkpoints (`model.joblib`, `model.pt`) and flow-level prediction arrays (`predictions.npz`, containing full $(N,)$ arrays for `graph_id`, `edge_id`, `y_true`, `y_pred`, and `y_proba`) are preserved locally on the primary training machine in `results/`. For paired flow-by-flow statistical tests against E-GraphSAGE, teammates can copy `results/` directly or download the checkpoint bundle from shared storage.
