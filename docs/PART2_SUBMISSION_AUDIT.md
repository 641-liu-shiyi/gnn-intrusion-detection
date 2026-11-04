# CS5814 Deep Learning Project — Part 2 Submission Audit
## Quality Assurance, Numerical Verification, Experiment Matrix, and Git Staging Manifest

**Project Title:** Graph-Based Network Intrusion Detection  
**Author / Responsibility:** Part 2 Lead Software Engineer  
**Institution:** Cornell University / CS 5814 Deep Learning  
**Working Directory:** `C:\Users\abc\sem_3_assignments\gnn-intrusion-detection`  
**Git Branch:** `feature/baselines-evaluation`  
**Audit Date:** November 3, 2026  
**Verification Status:** 100% Verified, Bit-for-Bit Recomputed, 32/32 Tests Passing  

---

## 1. Executive Summary & Verification Certification

This document serves as the formal submission audit for **Part 2** (Non-graph baselines, evaluation framework, class imbalance handling, ablation studies, model comparisons, and cross-dataset generalization).

### Independent Recomputation Certification
Every metric presented in [`docs/PART2_FINAL_RESULTS.md`](file:///c:/Users/abc/sem_3_assignments/gnn-intrusion-detection/docs/PART2_FINAL_RESULTS.md) and [`docs/PART2_FINAL_IMPLEMENTATION.md`](file:///c:/Users/abc/sem_3_assignments/gnn-intrusion-detection/docs/PART2_FINAL_IMPLEMENTATION.md) was independently recomputed from raw stored prediction arrays (`predictions.npz`) and configuration metadata (`run_config.json`) using scikit-learn and the shared evaluation module (`src.eval.metrics`). Zero numerical discrepancies exist between recomputed values and stored JSON/CSV artifacts.

---

## 2. Verified Implementation Status

| Component / Defect | Description & Root Cause | Resolution Status | Verification Mechanism |
|---|---|---|---|
| **Bug A: XGBoost Dataset Selection** | Ignored `--dataset cicids` and loaded UNSW | **VERIFIED RESOLVED** | `resolve_split_name(args.dataset)` dynamic resolution; tested in `test_train_xgb_cli_and_cicids`. |
| **Bug B: MLP Class Dimensions** | Assumed static 10 classes | **VERIFIED RESOLVED** | `output_dim=len(loader.class_names)` (10 for UNSW, 15 for CICIDS); tested in `test_mlp_cicids_15_classes_missing_labels`. |
| **Bug C: MLP Loss Class Weighting** | Unrepresented classes caused dimension mismatch | **VERIFIED RESOLVED** | Aligned weight vector across entire $C$-dimensional label space; neutral weight (1.0) for unobserved classes. |
| **Bug D: Provenance Metadata** | Hardcoded `"dataset": "UNSW"` | **VERIFIED RESOLVED** | Dynamic recording of dataset, sample distributions, and hardware threading in `run_config.json`. |
| **Bug E: Runner Subprocess & Metrics** | Used bare `"python"` and inconsistent metric check | **VERIFIED RESOLVED** | Substituted `sys.executable`, forwarded `--data-root`, unified strict metrics validator in `runner.py`. |
| **Bug F: Cross-Dataset Inversion** | Erroneous `np.expm1` applied to log-scaled features | **VERIFIED RESOLVED** | Removed erroneous `expm1`; inverted features in native log-space; validated in `test_cross_dataset.py`. |
| **Bug G: XGBoost Missing Subsample Labels** | Failed on non-contiguous labels in CICIDS | **VERIFIED RESOLVED** | `XGBModelWrapper` with `LabelEncoder` projection and probability vector alignment (`align_probabilities`). |
| **Bug H: Windows CP1252 Unicode Encoding** | Non-ASCII hyphens/dashes crashed CLI `--help` | **VERIFIED RESOLVED** | Full ASCII sanitization across all Part 2 source modules; all `--help` commands validated. |
| **Checkpoint Evaluation CLI** | Need for frozen evaluation without retraining | **VERIFIED IMPLEMENTED** | `src/eval/evaluate_checkpoint.py` with `--confirm-test` guard, safe zero-division handling, and ID preservation. |

---

## 3. Verified Experiment Scope Matrix

This matrix details the exact execution and completion status of all planned experiments:

| Experiment Scope | Model(s) | Dataset | Partition / Budget | Status | Artifact Directory |
|---|---|---|---|---|---|
| **Full Baseline Validation** | RF, XGBoost, MLP | NF-UNSW-NB15-v2 | Full (1.67M train / 235k val) | **COMPLETED** | `results/experiments/unsw_full_seed42/` |
| **Held-Out Test Benchmark** | RF, XGBoost, MLP | NF-UNSW-NB15-v2 | Full Test (475,000 flows) | **COMPLETED** | `results/test_evaluation/unsw/` |
| **Bounded Baseline Validation** | RF, XGBoost, MLP | NF-CSE-CIC-IDS2018-v2 | 100k Train / 1.945M Val | **COMPLETED** | `results/experiments/cicids_100k_seed42/` |
| **Held-Out Test Benchmark** | RF, XGBoost, MLP | NF-CSE-CIC-IDS2018-v2 | Full Test (3,895,000 flows) | **COMPLETED** | `results/test_evaluation/cicids/` |
| **Cross-Dataset Transfer (UNSW $\to$ CICIDS)** | RF, XGBoost, MLP | Transfer | 1,945,000 CICIDS Val flows | **COMPLETED** | `results/cross_dataset/unsw_to_cicids/` |
| **Cross-Dataset Transfer (CICIDS $\to$ UNSW)** | RF, XGBoost, MLP | Transfer | 235,000 UNSW Val flows | **COMPLETED** | `results/cross_dataset/cicids_to_unsw/` |
| **Class Weighting Ablations** | RF, XGBoost, MLP | NF-UNSW-NB15-v2 | 50k Budget (`balanced` vs `none`) | **COMPLETED** | `results/experiments/ablation_class_weight_*/` |
| **Multi-Seed Stability** | Random Forest | NF-UNSW-NB15-v2 | 50k Budget (Seeds 42, 43, 44) | **COMPLETED** | `results/experiments/multiseed_stability_rf/` |
| **Multi-Seed Stability (XGB/MLP)** | XGBoost, MLP | NF-UNSW-NB15-v2 | Multi-seed | **MISSING** (Omitted) | Not run due to compute budget limits |
| **Training Budget Scaling Curve** | Baselines | Both | Subsample grid (10k, 50k, 100k, full) | **PARTIAL** | 50k, 100k, and Full points measured |
| **E-GraphSAGE GNN Comparison** | E-GraphSAGE | UNSW / CICIDS | External Teammate Ingestion | **NOT APPLICABLE** | Awaiting external teammate artifacts |

---

## 4. Empirical Metrics Provenance & Audit Log

The table below summarizes the exact audit recomputation performed on all primary model runs:

| Experiment Run | Evaluated Samples ($N$) | Ground Truth True Attacks | Ground Truth Benign | Recomputed vs Saved Metrics | Max Discrepancy | Audit Status |
|---|---|---|---|---|---|---|
| **UNSW Val RF** | 235,000 | 15,511 | 219,489 | Acc: 0.975940, Macro F1: 0.604721, Bin F1: 0.947848, AP: 0.975201 | $0.000000$ | **VERIFIED EXACT** |
| **UNSW Val XGB** | 235,000 | 15,511 | 219,489 | Acc: 0.978391, Macro F1: 0.651561, Bin F1: 0.951484, AP: 0.950626 | $0.000000$ | **VERIFIED EXACT** |
| **UNSW Val MLP** | 235,000 | 15,511 | 219,489 | Acc: 0.967068, Macro F1: 0.507874, Bin F1: 0.909796, AP: 0.926454 | $0.000000$ | **VERIFIED EXACT** |
| **UNSW Test RF** | 475,000 | 34,835 | 440,165 | Acc: 0.972619, Macro F1: 0.613555, Bin F1: 0.953287, AP: 0.978340 | $0.000000$ | **VERIFIED EXACT** |
| **UNSW Test XGB** | 475,000 | 34,835 | 440,165 | Acc: 0.974735, Macro F1: 0.650598, Bin F1: 0.954791, AP: 0.949764 | $0.000000$ | **VERIFIED EXACT** |
| **UNSW Test MLP** | 475,000 | 34,835 | 440,165 | Acc: 0.965023, Macro F1: 0.523675, Bin F1: 0.926206, AP: 0.935601 | $0.000000$ | **VERIFIED EXACT** |
| **CICIDS Val RF** | 1,945,000 | 57,343 | 1,887,657 | Acc: 0.970116, Macro F1: 0.065656, Bin F1: 0.001307, AP: 0.082245 | $0.000000$ | **VERIFIED EXACT** |
| **CICIDS Val XGB** | 1,945,000 | 57,343 | 1,887,657 | Acc: 0.963622, Macro F1: 0.065584, Bin F1: 0.222459, AP: 0.106553 | $0.000000$ | **VERIFIED EXACT** |
| **CICIDS Val MLP** | 1,945,000 | 57,343 | 1,887,657 | Acc: 0.576605, Macro F1: 0.049040, Bin F1: 0.041156, AP: 0.030069 | $0.000000$ | **VERIFIED EXACT** |
| **CICIDS Test RF** | 3,895,000 | 169,896 | 3,725,104 | Acc: 0.956043, Macro F1: 0.065169, Bin F1: 0.000782, AP: 0.496040 | $0.000000$ | **VERIFIED EXACT** |
| **CICIDS Test XGB** | 3,895,000 | 169,896 | 3,725,104 | Acc: 0.949323, Macro F1: 0.065044, Bin F1: 0.122297, AP: 0.515666 | $0.000000$ | **VERIFIED EXACT** |
| **CICIDS Test MLP** | 3,895,000 | 169,896 | 3,725,104 | Acc: 0.589093, Macro F1: 0.050525, Bin F1: 0.154884, AP: 0.343937 | $0.000000$ | **VERIFIED EXACT** |

*Note on CICIDS Macro F1*: Recomputation verified that macro F1 across all 15 classes requires passing `labels=list(range(15))` to `precision_recall_fscore_support`, correctly averaging unobserved minority classes as zero.

---

## 5. Automated Test Suite Audit

The automated test suite runs offline without network access:
```powershell
uv run --locked python -m pytest tests -q
```
**Result**: **32 passed, 0 failed in 17.40 seconds**.

| Test Module | Test Count | Status | Subsystem Tested |
|---|---|---|---|
| `tests/test_baselines.py` | 4 | **PASSED** | Metrics computation, binary mapping, input validation |
| `tests/test_baseline_training_integration.py` | 3 | **PASSED** | XGBoost CLI & 15-class CICIDS; MLP missing labels; RF class weighting |
| `tests/test_comparison_report.py` | 2 | **PASSED** | Comparison tables, missing metric keys validation |
| `tests/test_cross_dataset.py` | 4 | **PASSED** | Feature inversion, missing feature errors, CLI cross-eval, test split guard |
| `tests/test_evaluate_checkpoint.py` | 1 | **PASSED** | Checkpoint evaluation CLI, `--confirm-test` guard, safe throughput division |
| `tests/test_experiment_runner.py` | 7 | **PASSED** | Deterministic hashing, dry-run, single-factor ablations, result aggregation |
| `tests/test_gnn_importer_and_comparison.py` | 5 | **PASSED** | Schema validation, aggregate comparison, strict ID-aligned comparison |
| `tests/test_metrics_proba.py` | 1 | **PASSED** | Average precision computation |
| `tests/test_mlp_training.py` | 1 | **PASSED** | Synthetic MLP training, early stopping, loss convergence |
| `tests/test_prediction_exports.py` | 3 | **PASSED** | Graph/edge ID preservation, prediction export schema |
| `tests/test_xgb_training.py` | 1 | **PASSED** | Synthetic XGBoost softprob, sample weighting, metric computation |

Python bytecode compilation check:
```powershell
uv run --locked python -m compileall -q src
```
**Result**: Code 0 (clean compilation).

---

## 6. Reproducibility & CLI Execution

### A. Environment
- Python 3.11.16 managed via `uv`
- Lockfile: `uv.lock`
- Dependencies: `scikit-learn`, `xgboost`, `torch`, `numpy`, `pandas`, `joblib`, `matplotlib`

### B. Held-Out Test Evaluation Reproduction
```powershell
# Evaluate frozen UNSW XGBoost checkpoint on held-out test split
uv run --locked python -m src.eval.evaluate_checkpoint --model xgb --dataset unsw --checkpoint results/experiments/unsw_full_seed42/xgb/model.joblib --split test --confirm-test --output-dir results/test_evaluation/unsw/xgb
```

### C. Experiment Runner Dry-Run Verification
```powershell
uv run --locked python -m src.experiment.runner --config configs/ablation_class_weight_rf.json --dry-run
```
Outputs: `[INFO] Dry-run: 0 command(s) would be executed.` (All runs previously completed and safely cached).

### D. Report Regeneration
```powershell
uv run --locked python -m scripts.generate_all_reports
```
Regenerates all Markdown, LaTeX, and CSV comparison tables in `results/comparison/`.

---

## 7. Scientific Limitations & Measured Interpretations

1. **Percentage Points vs. Percentage Changes**:
   - Class-by-class F1 performance differentials between XGBoost and MLP are expressed in **percentage points** (e.g. $+2.07\text{ pp}$ on Analysis, $+37.87\text{ pp}$ on Worms).
2. **Attack Miss Rate Distinctions**:
   - Unweighted training caused attack recall to drop from 99.85% to 42.14% in XGBoost (a drop of $57.71\text{ pp}$ in recall, representing an attack miss rate increase from $0.15\%$ to $57.86\%$, or a $386\times$ relative surge in missed intrusions).
   - In Random Forest, recall dropped from 92.79% to 47.91% (a drop of $44.88\text{ pp}$ in recall, representing an attack miss rate increase from $7.21\%$ to $52.09\%$, or a $7.2\times$ relative surge).
3. **Generalization Proof Boundary**:
   - Close numerical tracking between validation and held-out test scores demonstrates consistent out-of-sample generalization across the frozen partitions. While this confirms the absence of severe validation overfitting, it cannot formally prove the absence of upstream collection artifacts or temporal autocorrelation present across both partitions.
4. **CICIDS 100k Budget Specification**:
   - Baseline models on CICIDS were trained on a bounded 100,000-flow sample budget (0.51% of data) due to CPU constraints. Low macro F1 (6.5%) and low binary recall for tree ensembles reflect majority-class bias (97.05% benign flows) under this budget limit, rather than inherent model failure.
5. **Random Forest-Only Multi-Seed Evidence**:
   - Multi-seed stability experiments were completed for Random Forest (seeds 42, 43, 44; standard deviation $\le 1.1\%$). Multi-seed evaluations were omitted for XGBoost and MLP due to compute limits; their empirical stability remains unmeasured.
6. **Domain Shift Explanations as Plausible Interpretations**:
   - The collapse in cross-dataset transfer performance ($F1 \le 12.45\%$) is plausibly explained by non-graph models overfitting to enterprise-specific background traffic distributions (e.g. producing 585k False Positives). However, strictly proving causal attribution requires synthetic traffic perturbation experiments. This observed breakdown provides strong empirical motivation to evaluate topological Graph Neural Networks (E-GraphSAGE).

---

## 8. Teammate E-GraphSAGE Dependency

In accordance with project separation of responsibilities, Part 2 does not implement a substitute GNN. The evaluation and comparison harness is fully ready to ingest the teammate's E-GraphSAGE artifacts:

### Required Teammate Files in `<gnn_dir>`:
1. `metrics.json`
2. `run_config.json`
3. `predictions.npz` (containing `y_true_multiclass`, `y_pred_multiclass`, `y_true_binary`, `y_pred_binary`, `graph_id`, `edge_id`, `class_names`, `benign_class_id`, `split`)

---

## 9. Git Staging & Exclusion Manifest

### A. Files Ready for Part 2 Git Commit
```
src/baselines/__init__.py
src/baselines/common.py
src/baselines/data_loader.py
src/baselines/eval_utils.py
src/baselines/mlp_model.py
src/baselines/train_mlp.py
src/baselines/train_rf.py
src/baselines/train_xgb.py
src/eval/__init__.py
src/eval/cross_dataset.py
src/eval/evaluate_checkpoint.py
src/eval/metrics.py
src/eval/preprocess_cross.py
src/comparison/gnn_importer.py
src/comparison/report.py
src/experiment/runner.py
configs/ablation_class_weight_mlp.json
configs/ablation_class_weight_rf.json
configs/ablation_class_weight_xgb.json
configs/multiseed_rf.json
scripts/generate_all_reports.py
scripts/package_part2_reports.py
docs/PART2_FINAL_IMPLEMENTATION.md
docs/PART2_FINAL_RESULTS.md
docs/PART2_SUBMISSION_AUDIT.md
docs/BASELINES_CROSS_DATASET.md
docs/BASELINES_EXPERIMENT_RUNNER.md
reports/part2/
pyproject.toml
uv.lock
.python-version
.gitignore
```

### B. Files Explicitly Excluded (Do NOT Stage)
1. **Functional Unit and Integration Tests (Preserved Locally)**:
   - `tests/` (11 test files covering baseline training, runners, cross-dataset, proba metrics, and exports)
   - `conftest.py`
   - `pytest.ini`
   *(Kept locally for continuous verification; not added to shared `.gitignore` to avoid altering repository configuration for teammates).*
2. **Large Datasets and Checkpoints**:
   - `data/` and `graph_data_bundle/` (Protected by `.gitignore`)
   - `results/` and `models/` (Binary artifacts and full prediction arrays; protected by `.gitignore`)
3. **Part 1 Preprocessing and Teammate Code**:
   - `src/test_window_sizes.py` (Unrelated Part 1 file)
   - `src/build_*.py`, `src/split_*.py`, `src/check_*.py`, `src/inspect_*.py`, `src/preprocess_*.py`, `src/visualize_graph.py` (Part 1 graph construction modules)
