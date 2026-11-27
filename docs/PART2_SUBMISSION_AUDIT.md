# CS5814 Deep Learning Project — Part 2 Submission Audit
## Quality Assurance, Numerical Verification, Experiment Matrix, and Git Staging Manifest

**Project Title:** Graph-Based Network Intrusion Detection  
**Author / Responsibility:** Part 2 Lead Software Engineer  
**Institution:** Cornell University / CS 5814 Deep Learning  
**Git Branch:** `feature/cicids-episode-baselines`  
**Verification Status:** Verified, Bit-for-Bit Recomputed (`0.00000000` discrepancy), 32/32 Part 2 Tests Passing

---

## 1. Executive Summary & Verification Certification

This document serves as the formal submission audit for **Part 2** (Non-graph baselines, evaluation framework, class imbalance handling, ablation studies, model comparisons, and cross-dataset generalization), updated for Shiyi's corrected CICIDS episode-aware split (`graphs_cicids_episode` / `features_cicids_episode`).

### Independent Recomputation Certification
Every metric presented in [`docs/PART2_FINAL_RESULTS.md`](file:///c:/Users/abc/sem_3_assignments/gnn-intrusion-detection/docs/PART2_FINAL_RESULTS.md) and [`reports/part2/CICIDS_EPISODE_RESULTS.md`](file:///c:/Users/abc/sem_3_assignments/gnn-intrusion-detection/reports/part2/CICIDS_EPISODE_RESULTS.md) was independently recomputed from raw stored prediction arrays (`predictions.npz`) via `scripts/audit_full_experiments.py` using `src.eval.metrics.compute_metrics` and `compute_proba_metrics`. Max scalar discrepancy across all 24 audited runs (UNSW val/test, CICIDS episode val/test, and 12 cross-dataset runs) is **`0.00000000`**, with **`0` duplicate `(graph_id, edge_id)` pairs** and **100% exact confusion matrix agreement**.

---

## 2. Verified Experiment Scope Matrix

| Experiment Scope | Model(s) | Dataset | Partition / Budget | Status | Artifact Directory |
|---|---|---|---|---|---|
| **Full Baseline Validation (UNSW)** | RF, XGBoost, MLP | NF-UNSW-NB15-v2 | Full (1.67M train / 235k val) | **PRESERVED** | `results/experiments/unsw_full_seed42/` |
| **Held-Out Test Benchmark (UNSW)** | RF, XGBoost, MLP | NF-UNSW-NB15-v2 | Full Test (475,000 flows) | **PRESERVED** | `results/test_evaluation/unsw/` |
| **Bounded Baseline Validation (CICIDS Episode)** | RF, XGBoost, MLP | NF-CSE-CIC-IDS2018-v2 (Episode) | 100k Train / 1.71M Val (342 graphs) | **COMPLETED** | `results/experiments/cicids_episode_100k_seed42/` |
| **Held-Out Test Benchmark (CICIDS Episode)** | RF, XGBoost, MLP | NF-CSE-CIC-IDS2018-v2 (Episode) | Full Test (4,475,000 flows, 895 graphs) | **COMPLETED** | `results/test_evaluation/cicids_episode/` |
| **Class Weighting Ablations (UNSW)** | RF, XGBoost, MLP | NF-UNSW-NB15-v2 | 50k Budget (`balanced` vs `none`) | **PRESERVED** | `results/experiments/ablation_class_weight_*/` |
| **Class Weighting Ablations (CICIDS Episode)** | RF, XGBoost, MLP | NF-CSE-CIC-IDS2018-v2 (Episode) | 100k Budget (`balanced` vs `none`) | **COMPLETED** | `results/experiments/ablation_class_weight_cicids_*/` |
| **Multi-Seed Stability (RF)** | Random Forest | UNSW (50k) & CICIDS Episode (100k) | Seeds `42, 43, 44` | **COMPLETED** | `results/experiments/multiseed_stability_{rf,cicids_rf}/` |
| **Cross-Dataset Transfer (UNSW -> CICIDS Episode)** | RF, XGBoost, MLP | Transfer | Val (1.71M) & Test (4.475M) | **COMPLETED** | `results/cross_dataset_episode/unsw_to_cicids_{val,test}/` |
| **Cross-Dataset Transfer (CICIDS Episode -> UNSW)** | RF, XGBoost, MLP | Transfer | Val (235k) & Test (475k) | **COMPLETED** | `results/cross_dataset_episode/cicids_to_unsw_{val,test}/` |
| **E-GraphSAGE Comparison (CICIDS Episode)** | E-GraphSAGE (+Attn) | NF-CSE-CIC-IDS2018-v2 (Episode) | Reference 3-seed full-train results | **COMPLETED** | `member3/results_cicids_episode/` |

---

## 3. Empirical Metrics Provenance & Audit Log

| Experiment Run | Evaluated Samples ($N$) | True Attacks | True Benign | Recomputed Metrics | Max Discrepancy | Audit Status |
|---|---:|---:|---:|---|---:|---|
| **UNSW Val RF** | 235,000 | 15,511 | 219,489 | Acc: 0.9759, Macro F1: 0.6047, Bin F1: 0.9478 | `0.00000000` | **VERIFIED EXACT** |
| **UNSW Val XGB** | 235,000 | 15,511 | 219,489 | Acc: 0.9784, Macro F1: 0.6516, Bin F1: 0.9515 | `0.00000000` | **VERIFIED EXACT** |
| **UNSW Val MLP** | 235,000 | 15,511 | 219,489 | Acc: 0.9671, Macro F1: 0.5079, Bin F1: 0.9098 | `0.00000000` | **VERIFIED EXACT** |
| **UNSW Test RF** | 475,000 | 34,835 | 440,165 | Acc: 0.9726, Macro F1: 0.6136, Bin F1: 0.9533 | `0.00000000` | **VERIFIED EXACT** |
| **UNSW Test XGB** | 475,000 | 34,835 | 440,165 | Acc: 0.9747, Macro F1: 0.6506, Bin F1: 0.9548 | `0.00000000` | **VERIFIED EXACT** |
| **UNSW Test MLP** | 475,000 | 34,835 | 440,165 | Acc: 0.9650, Macro F1: 0.5237, Bin F1: 0.9262 | `0.00000000` | **VERIFIED EXACT** |
| **CICIDS Ep Val RF** | 1,710,000 | 209,425 | 1,500,575 | Acc: 0.7546, Macro F1: 0.5994, Bin F1: 0.4992, AP: 0.8785 | `0.00000000` | **VERIFIED EXACT** |
| **CICIDS Ep Val XGB** | 1,710,000 | 209,425 | 1,500,575 | Acc: 0.7230, Macro F1: 0.5279, Bin F1: 0.4730, AP: 0.9595 | `0.00000000` | **VERIFIED EXACT** |
| **CICIDS Ep Val MLP** | 1,710,000 | 209,425 | 1,500,575 | Acc: 0.5711, Macro F1: 0.5048, Bin F1: 0.3657, AP: 0.9188 | `0.00000000` | **VERIFIED EXACT** |
| **CICIDS Ep Test RF** | 4,475,000 | 418,823 | 4,056,177 | Acc: 0.7720, Macro F1: 0.5590, Bin F1: 0.4392, AP: 0.8578 | `0.00000000` | **VERIFIED EXACT** |
| **CICIDS Ep Test XGB** | 4,475,000 | 418,823 | 4,056,177 | Acc: 0.7432, Macro F1: 0.5093, Bin F1: 0.4191, AP: 0.9435 | `0.00000000` | **VERIFIED EXACT** |
| **CICIDS Ep Test MLP** | 4,475,000 | 418,823 | 4,056,177 | Acc: 0.5966, Macro F1: 0.4531, Bin F1: 0.3170, AP: 0.9449 | `0.00000000` | **VERIFIED EXACT** |

---

## 4. Automated Test Suite & Compilation

- `.\.venv\Scripts\python.exe -m compileall -q src scripts`: exit code `0`
- `.\.venv\Scripts\python.exe -m pytest tests --ignore=tests/test_explain.py -q`: **32 passed in 20.05s**
- Note: `tests/test_explain.py` (upstream LLM explanation suite) has 2 Windows-specific `WinError 10106` failures when stripping `SYSTEMROOT` via `env={"PATH": "/usr/bin:/bin"}`; all Part 2 tests pass 32/32.

---

## 5. Git Staging & Exclusion Manifest

### Included in `feature/cicids-episode-baselines` Commit
- `src/baselines/{common,data_loader,train_rf,train_xgb,train_mlp}.py`
- `src/eval/{cross_dataset,evaluate_checkpoint}.py`
- `src/experiment/runner.py`
- `configs/{ablation_class_weight_cicids_rf,ablation_class_weight_cicids_xgb,ablation_class_weight_cicids_mlp,multiseed_cicids_rf}.json`
- `scripts/{audit_full_experiments,audit_physical_data,generate_all_reports,package_part2_reports}.py`
- `reports/part2/` (corrected-split tables, LaTeX tables, per-class CSVs, figures, Markdown summaries, and archived chronological outputs in `archive_chronological_split/`)
- `docs/{PART2_FINAL_RESULTS,PART2_SUBMISSION_AUDIT}.md`

### Explicitly Excluded (Never Committed)
- `data/`, `graph_data_bundle/`, `results/` (`.gitignore`)
- Local `tests/test_*.py` (11 Part 2 tests), `conftest.py`, `pytest.ini`
- Unrelated Part 1 local modifications (`src/test_window_sizes.py` and line-ending diffs)
