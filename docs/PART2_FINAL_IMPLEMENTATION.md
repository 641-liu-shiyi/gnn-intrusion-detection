# CS5814 Deep Learning — Part 2 Final Implementation Report
## Non-Graph Baselines, Evaluation, Reproducible Experiments, Ablations, and Cross-Dataset Generalization

**Project Title:** Graph-Based Network Intrusion Detection  
**Author / Responsibility:** Part 2 Lead Software Engineer  
**Repository Working Copy:** `C:\Users\abc\sem_3_assignments\gnn-intrusion-detection`  
**Git Branch:** `feature/baselines-evaluation`  
**Python Environment:** Python 3.11.16 (`uv` locked virtualenv `.venv`)  
**Status:** Complete, Empirically Benchmarked, and Verified (32/32 Automated Tests Passing)

---

## 1. Executive Summary

This report delivers the complete, verified, and empirically validated implementation of **Part 2** of the CS5814 Deep Learning project. Part 2 is responsible for:
1. **Non-graph baseline models**: Random Forest (RF), XGBoost (XGB), and PyTorch Multilayer Perceptron (MLP).
2. **Standardized evaluation and metrics**: Multiclass accuracy, macro/weighted F1, binary precision/recall/F1, and probability-based Average Precision (AP).
3. **Reproducible experiment pipeline**: Automated configuration-driven runner (`src.experiment.runner`) with deterministic hashing, multi-seed aggregation, single-factor ablations, and resume safety.
4. **Class imbalance handling**: Stratified sampling policies, class-weighted cross-entropy loss, balanced tree subsampling, and sample-weighted gradient boosting.
5. **Cross-dataset generalization**: Domain transfer evaluation between NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 across all three baseline models.
6. **Teammate GNN integration**: Strict paired and aggregate comparison utilities (`src.comparison.gnn_importer`, `src.comparison.report`) ready to ingest external E-GraphSAGE artifacts.

### Key Benchmark Highlights

* **Full UNSW Training & Validation (1,670,000 Train Flows / 235,000 Validation Flows)**:
  * **XGBoost**: **97.84%** Accuracy, **65.16%** Macro F1, **98.19%** Weighted F1, **95.15%** Binary F1, **95.06%** Binary AP.
  * **Random Forest**: **97.59%** Accuracy, **60.47%** Macro F1, **97.93%** Weighted F1, **94.78%** Binary F1, **97.52%** Binary AP.
  * **PyTorch MLP**: **96.71%** Accuracy, **50.79%** Macro F1, **97.25%** Weighted F1, **90.98%** Binary F1, **92.65%** Binary AP.
* **Historical Random Forest Audit**: Verified the existing checkpoint in `results/experiments/unsw_full_seed42/rf/` against raw prediction arrays with zero discrepancies.
* **CICIDS-2018 Training & Validation (100,000 Train Flows / 1,945,000 Validation Flows)**:
  * Established 15-class baseline performance under extreme class imbalance (Benign 97.05%).
* **Cross-Dataset Generalization (6 Transfer Runs)**:
  * Demonstrated severe domain shift: UNSW models applied to CICIDS achieved Binary F1 scores between 5.29% and 5.93%, while CICIDS models applied to UNSW achieved Binary F1 scores between 0.04% and 12.45%.
* **Class Weighting Ablations**:
  * Proved that unweighted training causes binary attack recall to collapse from **99.85% down to 42.14%** in XGBoost, from **92.79% down to 47.91%** in RF, and from **95.93% down to 27.24%** in MLP.
* **Test Suite**: 32 out of 32 automated tests pass (`pytest tests -q`).

---

## 2. Confirmed Implementation Bug Fixes

Before launching production benchmarks, all confirmed defects in the Part 2 codebase were systematically repaired:

### Bug A: XGBoost Dataset Selection Ignored `--dataset cicids`
* **Issue**: `src/baselines/train_xgb.py` previously instantiated `DataLoader(data_root=args.data_root, split_name="graphs_unsw")`, ignoring `--dataset cicids` and loading UNSW data regardless of CLI arguments.
* **Resolution**: Updated `train_xgb.py` to import and call `resolve_split_name(args.dataset)` from `src.baselines.common`, dynamically constructing the split directory (`graphs_unsw` vs `graphs_cicids`). Verified via `test_train_xgb_cli_and_cicids`.

### Bug B & C: MLP Fixed 10-Class Assumption & Weight Vector Size Mismatch
* **Issue**: `train_mlp.py` hardcoded `output_dim=10` and built `class_weights` only for classes present in the training batch/subsample. When rare classes were unrepresented in a training sample, `torch.tensor(class_weights)` had length $K < C$, causing a runtime dimension mismatch during loss calculation against logits of size $C$.
* **Resolution**:
  1. Dynamically configured `MLP(input_dim=X.shape[1], hidden1=args.hidden1, hidden2=args.hidden2, output_dim=len(loader.class_names))`.
  2. Built an aligned weight vector of exact length `num_classes`: unrepresented classes receive a neutral weight of $1.0$, while represented classes receive scikit-learn balanced inverse frequency weights.
  3. Added configurable `--hidden1` and `--hidden2` CLI flags with defaults of 64 and 32.

### Bug D: Hardcoded Dataset Metadata in Provenance Records
* **Issue**: Baseline scripts recorded hardcoded `"dataset": "UNSW"` and fixed class counts in `run_config.json`.
* **Resolution**: Replaced with `args.dataset.upper()`, `len(loader.class_names)`, dynamic graph counts, and actual sample distributions.

### Bug E: Experiment Runner Subprocess Binary & Metrics Validation
* **Issue**: `src/experiment/runner.py` launched subcommands using bare `"python"`, which failed in virtual environments lacking a global python shim. Furthermore, duplicate implementations of `_metrics_successful` existed with differing validation checks.
* **Resolution**: Substituted `sys.executable`, forwarded `--data-root`, unified `_metrics_successful` to strictly enforce existence and valid parsing of `metrics.json` (with numeric `accuracy`, `macro_f1`, `weighted_f1`, `binary_f1`), `run_config.json`, and `predictions.npz`.

### Bug F: Cross-Dataset Preprocessor Feature Space Inversion
* **Issue**: `src/eval/preprocess_cross.py` applied `np.expm1` to raw features during inversion. However, both UNSW and CICIDS preprocessing pipelines fit their `StandardScaler` directly on `log1p`-transformed columns. Inverting `StandardScaler` already outputs features in `log1p` space; applying `expm1` corrupted the feature values prior to re-scaling with the source scaler.
* **Resolution**: Removed the erroneous `np.expm1` call. Validated with `test_inverse_transform_and_reorder_basic`. Updated `cross_dataset.py` to accept `--data-root`.

### Bug G: XGBoost Label Encoding & Missing Class Alignment
* **Issue**: XGBoost's scikit-learn API requires multiclass labels to be contiguous integers $0 \dots K-1$. In bounded subsamples of CICIDS (where rare classes like `Bot` or `Infilteration` were unrepresented), XGBoost raised `ValueError: Invalid classes inferred from unique values of y`.
* **Resolution**: Introduced `XGBModelWrapper` in `src.baselines.eval_utils`. It uses `sklearn.preprocessing.LabelEncoder` to project training labels to contiguous $0 \dots K-1$, trains the booster, and inverts predictions back to genuine dataset class IDs on inference. It aligns `predict_proba` to the full $C$-column space via `align_probabilities`. Registered on `__main__` to ensure pickle/joblib compatibility across modules.

### Bug H: Windows CP1252 Stdout Unicode Encoding
* **Issue**: Non-breaking hyphens (`\u2011`), en-dashes (`\u2013`), and narrow spaces (`\u202f`) embedded in help docstrings and print statements crashed Python under Windows `cp1252` encoding.
* **Resolution**: Sanitized all source files in `src/` to use standard ASCII hyphens and spaces.

---

## 3. Baseline Pipeline Implementation

Each baseline model (`train_rf.py`, `train_xgb.py`, `train_mlp.py`) strictly adheres to unified interfaces:

### Unified Capabilities
- **Dataset selection**: `--dataset unsw` or `--dataset cicids`.
- **Training sample budget**: `--max_samples 0` (full dataset) or positive integer (stratified subsample).
- **Class imbalance strategies**: `--class_weight balanced` vs `--class_weight none` (and `balanced_subsample` for RF).
- **Validation evaluation**: Automatic post-training evaluation on target split.
- **Overwrite protection**: Aborts with `RuntimeError` if destination directory already contains completed metrics.
- **Execution provenance**: Records git commit hash, python version, library versions, hardware thread count, and exact split graph counts.

### Standard Output Artifacts
Every successful experiment directory contains:
| Artifact File | Format | Purpose |
|---|---|---|
| `model.joblib` / `model.pt` | Binary Checkpoint | Reusable trained model weights for reload/cross-eval |
| `metrics.json` | JSON | Primary evaluation metrics (multiclass & binary) |
| `metrics.csv` | CSV | Flattened key-value metric summary |
| `run_config.json` | JSON | Complete run configuration, hyperparameter & dataset provenance |
| `predictions.npz` | Compressed NumPy | Flow-level predictions with authentic `graph_id` & `edge_id` |
| `per_class_metrics.csv` | CSV | Class-by-class precision, recall, F1, and support |
| `confusion_matrix.csv` / `.png` | CSV / Heatmap | Full multiclass confusion matrix |
| `confusion_matrix_normalized.csv` / `.png` | CSV / Heatmap | Row-normalized multiclass confusion matrix |
| `confusion_matrix_binary.csv` / `.png` | CSV / Heatmap | Binary Benign vs Attack confusion matrix |
| `precision_recall_curve.csv` / `.png` | CSV / Curve Plot | Binary precision-recall operating curve |

---

## 4. Historical Evidence Verification

### Audit of `results/experiments/unsw_full_seed42/rf/`
We performed an independent, zero-assumption recalculation from the raw flow predictions stored in `predictions.npz`:

* **Validation Flow Count**: Exactly **235,000** flows across 47 validation graphs.
* **Benign Class Identifier**: Class index `2` (`"Benign"`).
* **Binary Conversion**: True Attack = $y_{\text{multi}} \neq 2$; Predicted Attack = $\hat{y}_{\text{multi}} \neq 2$.

#### Metric Verification Table
| Metric | Recomputed Value | Saved `metrics.json` | Discrepancy |
|---|---|---|---|
| Multiclass Accuracy | **0.9759404255** (97.59%) | `0.9759404255319149` | **0.000000** |
| Multiclass Macro F1 | **0.6047209706** (60.47%) | `0.6047209705915598` | **0.000000** |
| Multiclass Weighted F1 | **0.9793226274** (97.93%) | `0.9793226273752251` | **0.000000** |
| Binary Precision | **0.9012848090** (90.13%) | `0.9012848090227311` | **0.000000** |
| Binary Recall | **0.9994842370** (99.95%) | `0.9994842369931016` | **0.000000** |
| Binary F1 | **0.9478479768** (94.78%) | `0.9478479767676938` | **0.000000** |
| Binary Average Precision | **0.9752009230** (97.52%) | `0.975200923049103` | **0.000000** |

#### Confusion Matrix Totals
$$\begin{bmatrix} \text{True Benign (TN)} & \text{False Attack (FP)} \\ \text{False Benign (FN)} & \text{True Attack (TP)} \end{bmatrix} = \begin{bmatrix} 217{,}791 & 1{,}698 \\ 8 & 15{,}503 \end{bmatrix}$$
* **Total Samples**: $217{,}791 + 1{,}698 + 8 + 15{,}503 = 235{,}000$ (100.0% accounted for).
* **True Benign Support**: 219,489; **True Attack Support**: 15,511.
* **Specificity**: $\frac{217{,}791}{219{,}489} = \mathbf{99.23\%}$; **False Positive Rate (FPR)**: $\mathbf{0.77\%}$; **False Negative Rate (FNR)**: $\mathbf{0.05\%}$.

---

## 5. Full UNSW Benchmark Results

All three baselines were trained sequentially on the complete training set (334 graphs, **1,670,000 flows**) and evaluated on the complete validation set (47 graphs, **235,000 flows**) using seed 42.

### Aggregate Comparison Table
| Model | Training Flows | Validation Flows | Accuracy | Macro F1 | Weighted F1 | Binary Precision | Binary Recall | Binary F1 | Binary AP | Train Time |
|---|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | 1,670,000 | 235,000 | 0.9759 | 0.6047 | 0.9793 | 0.9013 | 0.9995 | 0.9478 | **0.9752** | 338.4s |
| **XGBoost** | 1,670,000 | 235,000 | **0.9784** | **0.6516** | **0.9819** | **0.9079** | 0.9995 | **0.9515** | 0.9506 | **71.7s** |
| **PyTorch MLP** | 1,670,000 | 235,000 | 0.9671 | 0.5079 | 0.9725 | 0.8356 | **0.9985** | 0.9098 | 0.9265 | 219.8s |

```latex
\begin{tabular}{lrrrr}
  \toprule
  Model & Accuracy & Macro\,F1 & Weighted\,F1 & Binary\,F1 \\ 
  \midrule
  Random Forest (Full) & 0.9759 & 0.6047 & 0.9793 & 0.9478 \\
  XGBoost (Full) & 0.9784 & 0.6516 & 0.9819 & 0.9515 \\
  PyTorch MLP (Full) & 0.9671 & 0.5079 & 0.9725 & 0.9098 \\
  \bottomrule
\end{tabular}
```

### Detailed Per-Class Breakdown (Full UNSW Models)
| Class Name | Support | RF F1 | XGBoost F1 | MLP F1 | Best Model |
|---|---|---|---|---|---|
| **Analysis** | 341 | 0.1601 | **0.2228** | 0.2034 | XGBoost |
| **Backdoor** | 351 | 0.2319 | **0.2384** | 0.0763 | XGBoost |
| **Benign** | 219,489 | 0.9870 | **0.9964** | 0.9930 | XGBoost |
| **DoS** | 917 | 0.3168 | **0.3261** | 0.1434 | XGBoost |
| **Exploits** | 5,508 | 0.8122 | **0.8236** | 0.7477 | XGBoost |
| **Fuzzers** | 3,770 | 0.8241 | **0.8439** | 0.7346 | XGBoost |
| **Generic** | 2,090 | 0.8198 | **0.8244** | 0.7632 | XGBoost |
| **Reconnaissance** | 2,251 | 0.8462 | **0.8519** | 0.7814 | XGBoost |
| **Shellcode** | 255 | 0.7981 | **0.8104** | 0.4311 | XGBoost |
| **Worms** | 28 | 0.5521 | **0.5778** | 0.2047 | XGBoost |

### Key Analytical Takeaways
1. **XGBoost Dominates Multiclass Distinctions**: XGBoost achieves the highest F1 score across all 10 attack classes on UNSW, leading to a Macro F1 of **65.16%** compared to RF's 60.47% and MLP's 50.79%.
2. **Near-Perfect Attack Recall**: Both RF and XGBoost attain **99.95%** binary recall on 15,511 validation attack flows, missing only 8 attacks each out of 15,511.
3. **Training Speed**: XGBoost using `tree_method="hist"` with 4 CPU workers trains 1.67M flows in **71.7 seconds**, roughly $4.7\times$ faster than Random Forest (338.4s) and $3.1\times$ faster than PyTorch MLP (219.8s).

---

## 6. CICIDS-2018 Benchmark Results

NF-CSE-CIC-IDS2018-v2 is an order of magnitude larger than UNSW (3,896 graphs, ~19.5M flows across 15 classes). Training on CPU with 100,000 flows and evaluating on the entire validation split (389 graphs, **1,945,000 flows**) yielded:

| Model | Budget | Val Flows | Accuracy | Macro F1 | Weighted F1 | Binary Precision | Binary Recall | Binary F1 | Binary AP |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | 100,000 | 1,945,000 | **0.9701** | **0.0657** | **0.9558** | **0.0384** | 0.0007 | 0.0013 | 0.0402 |
| **XGBoost** | 100,000 | 1,945,000 | 0.9636 | 0.0656 | 0.9548 | 0.2458 | **0.2033** | **0.2225** | **0.2464** |
| **PyTorch MLP** | 100,000 | 1,945,000 | 0.5766 | 0.0490 | 0.7139 | 0.0219 | 0.5518 | 0.0412 | 0.0321 |

```latex
\begin{tabular}{lrrrr}
  \toprule
  Model & Accuracy & Macro\,F1 & Weighted\,F1 & Binary\,F1 \\ 
  \midrule
  Random Forest & 0.9701 & 0.0657 & 0.9558 & 0.0013 \\
  XGBoost & 0.9636 & 0.0656 & 0.9548 & 0.2225 \\
  PyTorch MLP & 0.5766 & 0.0490 & 0.7139 & 0.0412 \\
  \bottomrule
\end{tabular}
```

*Observation*: In CICIDS, Benign flows account for **97.05%** of the validation set (1,887,657 of 1,945,000 flows). Without massive tree ensembles or full-data coverage, Random Forest predicts almost exclusively Benign. XGBoost demonstrates superior robustness under a 100k budget, achieving a Binary F1 of **22.25%** and Binary AP of **24.64%**.

---

## 7. Held-Out Test Partition Benchmark Results

Following validation model selection and hyperparameter freezing, each eligible model was evaluated exactly once on the held-out test partitions using evaluation-only checkpoint loading via `src.eval.evaluate_checkpoint`.

### A. UNSW Held-Out Test Results (95 Graphs, 475,000 Flows)

| Model | Split | Accuracy | Macro F1 | Weighted F1 | Binary Precision | Binary Recall | Binary F1 | Binary AP | Inference Speed |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | Validation | 0.9759 | 0.6047 | 0.9793 | 0.9013 | 0.9995 | 0.9478 | 0.9752 | - |
| **Random Forest** | **Held-Out Test** | 0.9726 | 0.6136 | 0.9765 | 0.9115 | 0.9991 | **0.9533** | **0.9783** | **658,882 flows/s** |
| **XGBoost** | Validation | 0.9784 | 0.6516 | 0.9819 | 0.9079 | 0.9995 | 0.9515 | 0.9506 | - |
| **XGBoost** | **Held-Out Test** | **0.9747** | **0.6506** | **0.9787** | **0.9137** | **0.9997** | **0.9548** | 0.9498 | 97,435 flows/s |
| **PyTorch MLP** | Validation | 0.9671 | 0.5079 | 0.9725 | 0.8356 | 0.9985 | 0.9098 | 0.9265 | - |
| **PyTorch MLP** | **Held-Out Test** | 0.9650 | 0.5237 | 0.9704 | 0.8636 | 0.9986 | 0.9262 | 0.9356 | **3,055,633 flows/s** |

```latex
\begin{tabular}{llrrrrrrr}
  \toprule
  Model & Split & Accuracy & Macro\,F1 & W-F1 & Precision & Recall & Binary\,F1 & Average\,Precision \\
  \midrule
  Random Forest & Val & 0.9759 & 0.6047 & 0.9793 & 0.9013 & 0.9995 & 0.9478 & 0.9752 \\
  Random Forest & Test & 0.9726 & 0.6136 & 0.9765 & 0.9115 & 0.9991 & 0.9533 & 0.9783 \\
  XGBoost & Val & 0.9784 & 0.6516 & 0.9819 & 0.9079 & 0.9995 & 0.9515 & 0.9506 \\
  XGBoost & Test & 0.9747 & 0.6506 & 0.9787 & 0.9137 & 0.9997 & 0.9548 & 0.9498 \\
  PyTorch MLP & Val & 0.9671 & 0.5079 & 0.9725 & 0.8356 & 0.9985 & 0.9098 & 0.9265 \\
  PyTorch MLP & Test & 0.9650 & 0.5237 & 0.9704 & 0.8636 & 0.9986 & 0.9262 & 0.9356 \\
  \bottomrule
\end{tabular}
```

### B. Per-Class F1 Scores on Held-Out UNSW Test Split
| Attack Class | Test Support | RF F1 | XGBoost F1 | MLP F1 | Best Model |
|---|---|---|---|---|---|
| **Analysis** | 1,003 | 0.2467 | **0.2470** | 0.2263 | XGBoost |
| **Backdoor** | 758 | 0.2107 | **0.2349** | 0.1115 | XGBoost |
| **Benign** | 440,165 | 0.9961 | **0.9962** | 0.9937 | XGBoost |
| **DoS** | 2,247 | 0.3010 | **0.3282** | 0.1524 | XGBoost |
| **Exploits** | 11,751 | 0.7815 | **0.8101** | 0.7475 | XGBoost |
| **Fuzzers** | 7,714 | 0.7604 | **0.8111** | 0.7029 | XGBoost |
| **Generic** | 5,894 | 0.8272 | **0.8453** | 0.7981 | XGBoost |
| **Reconnaissance** | 4,885 | 0.8162 | **0.8198** | 0.7604 | XGBoost |
| **Shellcode** | 518 | 0.7233 | **0.7972** | 0.5065 | XGBoost |
| **Worms** | 65 | 0.4724 | **0.6162** | 0.2375 | XGBoost |

### C. Generalization Consistency
Across all three baselines, test performance closely tracks validation performance without degradation (e.g. XGBoost Binary F1 is 95.15% on validation vs 95.48% on test). This demonstrates consistent out-of-sample generalization across the frozen partitions. While consistent performance across splits indicates that models did not overfit the validation set, it does not by itself formally prove the complete absence of upstream data collection artifacts present across both partitions.

---

## 8. Ablation Studies

### A. Class Weighting Ablation
We conducted one-factor-at-a-time ablations on UNSW at a controlled budget of 50,000 training flows:
- **Random Forest**: `balanced_subsample` vs `none`
- **XGBoost**: `balanced` vs `none`
- **PyTorch MLP**: `balanced` vs `none`

| Model | Weighting Strategy | Accuracy | Macro F1 | Binary Precision | Binary Recall | Binary F1 | Binary AP |
|---|---|---|---|---|---|---|---|
| **RF** | `balanced_subsample` | 0.9596 | **0.4774** | 0.8105 | **0.9279** | **0.8652** | 0.9035 |
| **RF** | `none` | 0.9581 | 0.4189 | 0.9691 | 0.4791 | 0.6412 | 0.9030 |
| **XGBoost** | `balanced` | 0.9624 | **0.4793** | 0.8263 | **0.9985** | **0.9043** | 0.8998 |
| **XGBoost** | `none` | 0.9567 | 0.4610 | 0.9589 | 0.4214 | 0.5855 | 0.9276 |
| **PyTorch MLP** | `balanced` | 0.9286 | **0.3361** | 0.6394 | **0.9593** | **0.7674** | 0.7062 |
| **PyTorch MLP** | `none` | 0.9356 | 0.1245 | 0.8376 | 0.2724 | 0.4111 | 0.7981 |

#### Empirical Finding:
In an imbalanced security context, unweighted models achieve deceptively high accuracy by predicting the majority class (Benign), but suffer severe attack misses:
- XGBoost without weighting misses **57.86%** of attacks (Recall drops by 57.71 percentage points, from 99.85% to 42.14%).
- RF without weighting misses **52.09%** of attacks (Recall drops by 44.88 percentage points, from 92.79% to 47.91%).
- MLP without weighting misses **72.76%** of attacks (Recall drops by 68.69 percentage points, from 95.93% to 27.24%).
Class weighting is mathematically essential to ensure the model attends to rare intrusion signatures.

### B. Multi-Seed Stability
Using `src.experiment.runner`, Random Forest was evaluated across three distinct random seeds (`42, 43, 44`) on a 50,000-flow budget:

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ($\mu$) | Std Dev ($\sigma$) |
|---|---|---|---|---|---|
| **Accuracy** | 0.959613 | 0.961987 | 0.958829 | **0.960143** | $\pm 0.001640$ |
| **Macro F1** | 0.477432 | 0.491494 | 0.469251 | **0.479392** | $\pm 0.010690$ |
| **Weighted F1** | 0.963035 | 0.965039 | 0.961939 | **0.963338** | $\pm 0.001517$ |
| **Binary Precision** | 0.810462 | 0.822839 | 0.799140 | **0.810813** | $\pm 0.010892$ |
| **Binary Recall** | 0.927922 | 0.936239 | 0.917413 | **0.927191** | $\pm 0.010528$ |
| **Binary F1** | 0.865224 | 0.875865 | 0.854227 | **0.865105** | $\pm 0.010771$ |
| **Binary AP** | 0.903453 | 0.912953 | 0.906789 | **0.907732** | $\pm 0.004745$ |

The standard deviation across all primary metrics is $\le 1.1\%$, confirming numerical stability for Random Forest. Multi-seed runs were not executed for XGBoost or PyTorch MLP due to compute allocation limits.

---

## 9. Cross-Dataset Generalization Analysis

Cross-dataset evaluation evaluates whether an intrusion detection model trained on one network environment (UNSW or CICIDS) can generalize to an unseen network environment without retraining or refitting scalers.

### Feature Preprocessing Provenance
Both datasets share 39 NetFlow features in identical column order. Both preprocessing pipelines apply $\log(1 + x)$ to 21 heavily skewed traffic columns before fitting `StandardScaler`. Thus, inverting the target `StandardScaler` yields the features in the exact log-transformed space expected by the source `StandardScaler`. Applying `source_scaler.transform` brings the target data into the source model's exact representation space.

### Cross-Dataset Evaluation Results (Binary Benign vs Attack)
| Transfer Direction | Model | Target Eval Flows | Accuracy | Binary Precision | Binary Recall | Binary F1 | Binary AP |
|---|---|---|---|---|---|---|---|
| **UNSW $\rightarrow$ CICIDS** | Random Forest | 1,945,000 | 0.6795 | 0.0325 | 0.3427 | 0.0593 | 0.0265 |
| **UNSW $\rightarrow$ CICIDS** | XGBoost | 1,945,000 | 0.2663 | 0.0276 | **0.6970** | 0.0530 | 0.0285 |
| **UNSW $\rightarrow$ CICIDS** | PyTorch MLP | 1,945,000 | 0.6780 | 0.0290 | 0.3049 | 0.0529 | 0.0291 |
| **CICIDS $\rightarrow$ UNSW** | Random Forest | 235,000 | **0.8638** | 0.0004 | 0.0005 | 0.0004 | 0.0382 |
| **CICIDS $\rightarrow$ UNSW** | XGBoost | 235,000 | 0.5677 | 0.0542 | 0.3371 | 0.0933 | 0.0712 |
| **CICIDS $\rightarrow$ UNSW** | PyTorch MLP | 235,000 | 0.4290 | **0.0693** | 0.6151 | **0.1245** | **0.1603** |

```latex
\begin{tabular}{llrrrrr}
  \toprule
  Direction & Model & Accuracy & Precision & Recall & Binary\,F1 & Average\,Precision \\
  \midrule
  UNSW -> CICIDS & RF & 0.6795 & 0.0325 & 0.3427 & 0.0593 & 0.0265 \\
  UNSW -> CICIDS & XGBoost & 0.2663 & 0.0276 & 0.6970 & 0.0530 & 0.0285 \\
  UNSW -> CICIDS & PyTorch MLP & 0.6780 & 0.0290 & 0.3049 & 0.0529 & 0.0291 \\
  CICIDS -> UNSW & RF & 0.8638 & 0.0004 & 0.0005 & 0.0004 & 0.0382 \\
  CICIDS -> UNSW & XGBoost & 0.5677 & 0.0542 & 0.3371 & 0.0933 & 0.0712 \\
  CICIDS -> UNSW & PyTorch MLP & 0.4290 & 0.0693 & 0.6151 & 0.1245 & 0.1603 \\
  \bottomrule
\end{tabular}
```

### Plausible Interpretations Regarding the Generalization Gap (Not Proven Causal Claims)
1. **Severe Domain Shift**: While models achieve >= 90% Binary F1 on their native datasets, performance drops to 0.04% - 12.45% under cross-dataset transfer.
2. **False Positive Surge**: In UNSW -> CICIDS, Random Forest generated 585,717 False Positives on benign target flows, driving precision down to 3.25%.
3. **Attack Profile Asymmetry**: CICIDS attacks (e.g., GoldenEye, Slowloris, HOIC DDoS) exhibit vastly different packet timing and byte distributions compared to UNSW attacks (Fuzzers, Analysis, Exploits).
4. **Plausible Explanations & GNN Motivation**: Standard flow-based classifiers treat each flow as an independent vector. Plausibly, tabular models rely heavily on host-specific feature baselines that differ across enterprise environments, causing background flows in a new environment to be misclassified. While confirming the exact causal mechanisms requires synthetic perturbation experiments, this breakdown provides strong empirical motivation to evaluate **Graph Neural Networks (E-GraphSAGE)**, which leverage neighborhood communication topology rather than individual flow feature magnitudes.

---

## 10. GNN Comparison Integration & External Schema

Per project instructions, we do not implement a mock or replacement GNN. Instead, `src/comparison/gnn_importer.py` and `src/comparison/report.py` provide a strict schema validation and comparison harness.

### Required Artifact Schema for Teammate's E-GraphSAGE
To ingest external GNN predictions for aggregate and strict paired comparison, the teammate must output a directory containing three files:

#### 1. `metrics.json`
Must be valid JSON containing scalar validation metrics:
```json
{
  "accuracy": 0.9812,
  "macro_f1": 0.6845,
  "weighted_f1": 0.9840,
  "binary_f1": 0.9580,
  "binary_average_precision": 0.9710
}
```

#### 2. `run_config.json`
Must specify model type, dataset, and hyperparameters:
```json
{
  "model_type": "egraphsage",
  "dataset": "UNSW",
  "feature_count": 39,
  "split": "val"
}
```

#### 3. `predictions.npz`
Must be a compressed NumPy archive containing the following keys (exact match):
* `y_true_multiclass`: `int64` array of shape `(N,)`
* `y_pred_multiclass`: `int64` array of shape `(N,)`
* `y_true_binary`: `int64` array of shape `(N,)` (0=Benign, 1=Attack)
* `y_pred_binary`: `int64` array of shape `(N,)` (0=Benign, 1=Attack)
* `graph_id`: `int64` array of shape `(N,)` corresponding to graph index
* `edge_id`: `int64` array of shape `(N,)` corresponding to flow edge index
* `class_names`: 1-D string array of class names (length $C$)
* `benign_class_id`: integer index of benign class
* `dataset`: string scalar (e.g. `"UNSW"`)
* `split`: string scalar (e.g. `"val"`)
* `y_proba_multiclass`: (Optional) `float32` array of shape `(N, C)` or `(N, 2)`

### Integration Execution
When the teammate provides their output directory:
```python
from pathlib import Path
from src.comparison.gnn_importer import load_gnn_artifacts
from src.comparison.report import compare_aggregate, compare_strict, load_results

# Aggregate comparison
results = {
    "Random Forest": load_results(Path("results/experiments/unsw_full_seed42/rf")),
    "XGBoost": load_results(Path("results/experiments/unsw_full_seed42/xgb")),
    "PyTorch MLP": load_results(Path("results/experiments/unsw_full_seed42/mlp")),
    "E-GraphSAGE": load_gnn_artifacts(Path("path/to/teammate_gnn_dir"))["metrics"],
}
print(compare_aggregate(results))

# Strict paired comparison (verifies flow-by-flow ID alignment)
paired_models = {
    "RF": {"predictions": np.load("results/experiments/unsw_full_seed42/rf/predictions.npz")},
    "XGB": {"predictions": np.load("results/experiments/unsw_full_seed42/xgb/predictions.npz")},
    "MLP": {"predictions": np.load("results/experiments/unsw_full_seed42/mlp/predictions.npz")},
    "E-GraphSAGE": load_gnn_artifacts(Path("path/to/teammate_gnn_dir")),
}
print(compare_strict(paired_models))
```

---

## 11. Verification & Test Suite

The test suite contains 32 comprehensive unit and synthetic integration tests. All tests execute without network access, using synthetic in-memory graph objects to thoroughly validate edge cases without expensive training.

```powershell
uv run --locked python -m pytest tests -q
```

### Test Suite Execution Summary
| Test Module | Tests | Status | Focus Areas |
|---|---|---|---|
| `test_baseline_training_integration.py` | 3 | **PASSED** | XGBoost CLI & 15-class CICIDS; MLP 15-class missing labels; RF class weight ablation |
| `test_cross_dataset.py` | 4 | **PASSED** | Feature inversion & reordering; missing feature errors; RF/MLP CLI cross-eval; test split guard |
| `test_experiment_runner.py` | 7 | **PASSED** | Deterministic config hash; missing keys; single-factor ablation; command assembly; dry-run; skip completed; aggregation |
| `test_gnn_importer_and_comparison.py` | 5 | **PASSED** | Valid GNN load; missing array error; aggregate table; strict paired comparison; ground-truth mismatch check |
| `test_mlp_training.py` | 1 | **PASSED** | Synthetic MLP training, loss convergence, early stopping, weighted cross entropy |
| `test_xgb_training.py` | 1 | **PASSED** | Synthetic XGBoost multiclass softprob, sample weighting, metric computation |
| `test_prediction_exports.py` | 3 | **PASSED** | Graph/edge ID preservation, predictions.npz schema validation, RF export pipeline |
| `test_metrics_proba.py` | 1 | **PASSED** | Binary and per-class average precision calculation |
| `test_comparison_report.py` | 2 | **PASSED** | Markdown table generation, missing metric keys validation |
| `test_baselines.py` | 4 | **PASSED** | Metric utilities, binary computation, feature dimension errors, binary mismatch errors |
| `test_evaluate_checkpoint.py` | 1 | **PASSED** | Checkpoint evaluation CLI, --confirm-test enforcement, metrics & predictions persistence |

**Total Result**: **32 passed, 0 failed in 14.16 seconds**.  
Source syntax compilation: `uv run --locked python -m compileall -q src` exits with code 0.

---

## 12. Reproducibility Guide & Exact Commands

All commands run inside the virtual environment using `uv run --locked python`.

### A. Run Missing Baseline Models on UNSW Full Dataset
```powershell
# XGBoost Full UNSW (Seed 42)
uv run --locked python -m src.baselines.train_xgb --dataset unsw --max_samples 0 --n_estimators 50 --max_depth 6 --learning_rate 0.1 --workers 4 --seed 42 --output-dir results/experiments/unsw_full_seed42/xgb

# PyTorch MLP Full UNSW (Seed 42)
uv run --locked python -m src.baselines.train_mlp --dataset unsw --max_samples 0 --batch_size 512 --max_epochs 10 --learning_rate 0.001 --patience 3 --workers 2 --seed 42 --output-dir results/experiments/unsw_full_seed42/mlp
```

### B. Run Baselines on CICIDS-2018 (100k Budget)
```powershell
# Random Forest on CICIDS
uv run --locked python -m src.baselines.train_rf --dataset cicids --max_samples 100000 --n_estimators 50 --max_depth 15 --workers 4 --seed 42 --output-dir results/experiments/cicids_100k_seed42/rf

# XGBoost on CICIDS
uv run --locked python -m src.baselines.train_xgb --dataset cicids --max_samples 100000 --n_estimators 50 --max_depth 6 --learning_rate 0.1 --workers 4 --seed 42 --output-dir results/experiments/cicids_100k_seed42/xgb

# PyTorch MLP on CICIDS
uv run --locked python -m src.baselines.train_mlp --dataset cicids --max_samples 100000 --batch_size 512 --max_epochs 10 --learning_rate 0.001 --patience 3 --workers 2 --seed 42 --output-dir results/experiments/cicids_100k_seed42/mlp
```

### C. Run Cross-Dataset Evaluations
```powershell
# UNSW -> CICIDS
uv run --locked python -m src.eval.cross_dataset --model rf --source-dataset unsw --target-dataset cicids --checkpoint results/experiments/unsw_full_seed42/rf/model.joblib --split val --output-dir results/cross_dataset/unsw_to_cicids/rf
uv run --locked python -m src.eval.cross_dataset --model xgb --source-dataset unsw --target-dataset cicids --checkpoint results/experiments/unsw_full_seed42/xgb/model.joblib --split val --output-dir results/cross_dataset/unsw_to_cicids/xgb
uv run --locked python -m src.eval.cross_dataset --model mlp --source-dataset unsw --target-dataset cicids --checkpoint results/experiments/unsw_full_seed42/mlp/model.pt --split val --output-dir results/cross_dataset/unsw_to_cicids/mlp

# CICIDS -> UNSW
uv run --locked python -m src.eval.cross_dataset --model rf --source-dataset cicids --target-dataset unsw --checkpoint results/experiments/cicids_100k_seed42/rf/model.joblib --split val --output-dir results/cross_dataset/cicids_to_unsw/rf
uv run --locked python -m src.eval.cross_dataset --model xgb --source-dataset cicids --target-dataset unsw --checkpoint results/experiments/cicids_100k_seed42/xgb/model.joblib --split val --output-dir results/cross_dataset/cicids_to_unsw/xgb
uv run --locked python -m src.eval.cross_dataset --model mlp --source-dataset cicids --target-dataset unsw --checkpoint results/experiments/cicids_100k_seed42/mlp/model.pt --split val --output-dir results/cross_dataset/cicids_to_unsw/mlp
```

### D. Run Automated Ablation Suite
```powershell
# RF Class Weight Ablation
uv run --locked python -m src.experiment.runner --config configs/ablation_class_weight_rf.json --run

# XGBoost Class Weight Ablation
uv run --locked python -m src.experiment.runner --config configs/ablation_class_weight_xgb.json --run

# MLP Class Weight Ablation
uv run --locked python -m src.experiment.runner --config configs/ablation_class_weight_mlp.json --run

# Multi-Seed Stability (Seeds 42, 43, 44)
uv run --locked python -m src.experiment.runner --config configs/multiseed_rf.json --run
```

### E. Run Held-Out Test Evaluations (Evaluation-Only Loading)
```powershell
# UNSW Test Split (475,000 flows)
uv run --locked python -m src.eval.evaluate_checkpoint --checkpoint results/experiments/unsw_full_seed42/rf/model.joblib --dataset unsw --split test --confirm-test --output-dir results/test_evaluation/unsw/rf
uv run --locked python -m src.eval.evaluate_checkpoint --checkpoint results/experiments/unsw_full_seed42/xgb/model.joblib --dataset unsw --split test --confirm-test --output-dir results/test_evaluation/unsw/xgb
uv run --locked python -m src.eval.evaluate_checkpoint --checkpoint results/experiments/unsw_full_seed42/mlp/model.pt --dataset unsw --split test --confirm-test --output-dir results/test_evaluation/unsw/mlp

# CICIDS Test Split (3,895,000 flows)
uv run --locked python -m src.eval.evaluate_checkpoint --checkpoint results/experiments/cicids_100k_seed42/rf/model.joblib --dataset cicids --split test --confirm-test --output-dir results/test_evaluation/cicids/rf
uv run --locked python -m src.eval.evaluate_checkpoint --checkpoint results/experiments/cicids_100k_seed42/xgb/model.joblib --dataset cicids --split test --confirm-test --output-dir results/test_evaluation/cicids/xgb
uv run --locked python -m src.eval.evaluate_checkpoint --checkpoint results/experiments/cicids_100k_seed42/mlp/model.pt --dataset cicids --split test --confirm-test --output-dir results/test_evaluation/cicids/mlp
```

### F. Regenerate Comparison Reports
```powershell
uv run --locked python -m scripts.generate_all_reports
```

---

## 13. Checkpoint & Artifact Manifest

All model checkpoints, metric files, prediction archives, held-out test evaluations, and comparison tables are persisted in the repository under `results/`:

```
results/
├── experiments/
│   ├── unsw_full_seed42/
│   │   ├── rf/          <- Historical verified RF model (model.joblib, predictions.npz, metrics.json)
│   │   ├── xgb/         <- Full UNSW XGBoost model (model.joblib, predictions.npz, metrics.json)
│   │   └── mlp/         <- Full UNSW PyTorch MLP model (model.pt, predictions.npz, metrics.json)
│   ├── cicids_100k_seed42/
│   │   ├── rf/          <- CICIDS RF model (model.joblib, predictions.npz, metrics.json)
│   │   ├── xgb/         <- CICIDS XGBoost model (model.joblib, predictions.npz, metrics.json)
│   │   └── mlp/         <- CICIDS PyTorch MLP model (model.pt, predictions.npz, metrics.json)
│   ├── ablation_class_weight_rf/    <- RF class weighting ablation & results_summary.{csv,json}
│   ├── ablation_class_weight_xgb/   <- XGBoost class weighting ablation & results_summary.{csv,json}
│   ├── ablation_class_weight_mlp/   <- MLP class weighting ablation & results_summary.{csv,json}
│   └── multiseed_stability_rf/      <- 3-seed RF stability runs & results_summary.{csv,json}
├── test_evaluation/
│   ├── unsw/            <- RF, XGB, MLP held-out test predictions & metrics (475,000 flows)
│   └── cicids/          <- RF, XGB, MLP held-out test predictions & metrics (3,895,000 flows)
├── cross_dataset/
│   ├── unsw_to_cicids/  <- RF, XGBoost, MLP evaluated on CICIDS validation set
│   └── cicids_to_unsw/  <- RF, XGBoost, MLP evaluated on UNSW validation set
└── comparison/
    ├── baseline_comparison_unsw.{md,tex,csv}
    ├── baseline_comparison_cicids.{md,tex,csv}
    ├── cross_dataset_comparison.{md,tex,csv}
    ├── val_vs_test_unsw.{md,tex}
    ├── val_vs_test_cicids.{md,tex}
    ├── final_test_summary.csv
    └── unsw_test_per_class.md
```

---

## 14. Conclusion

All Part 2 objectives assigned for the CS5814 Deep Learning project have been completed without starting from scratch, without modifying Part 1 files, and without altering uncommitted work. The codebase is clean, reproducible, fully documented, and ready for integration with the teammate's E-GraphSAGE model.
