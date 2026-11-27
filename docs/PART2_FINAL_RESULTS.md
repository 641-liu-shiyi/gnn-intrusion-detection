# CS5814 Deep Learning Project — Part 2 Final Results Report
## Non-Graph Baselines, Evaluation Framework, Ablation Studies, Cross-Dataset Generalization, and CICIDS Episode-Split Update

**Project Title:** Graph-Based Network Intrusion Detection  
**Author / Responsibility:** Part 2 Lead Software Engineer  
**Institution:** Cornell University / CS 5814 Deep Learning  
**Repository Branch:** `feature/cicids-episode-baselines`  
**Execution Environment:** Python 3.11.16 (`uv` locked virtualenv `.venv`)  
**Artifact Status:** Empirically Validated and Reproducible (32/32 Part 2 Tests Passing)

---

## 1. Abstract

Network Intrusion Detection Systems (NIDS) operating on NetFlow records face extreme class imbalance, diverse attack families, and severe performance degradation under domain transfer across disparate enterprise networks. As Part 2 of the CS5814 Graph-Based Network Intrusion Detection project, this investigation develops, optimizes, and evaluates three non-graph baseline models—**Random Forest (RF)**, **XGBoost**, and a **PyTorch Multilayer Perceptron (MLP)**—across two industry-standard benchmarks: **NF-UNSW-NB15-v2** and **NF-CSE-CIC-IDS2018-v2** (using Shiyi's corrected episode-aware split `graphs_cicids_episode` / `features_cicids_episode`).

We freeze model architectures on validation partitions and conduct held-out test evaluations across **475,000 flows (UNSW)** and **4,475,000 flows (CICIDS episode split)**. On **UNSW**, XGBoost achieves the strongest overall baseline performance among the evaluated non-graph models with **97.47% test accuracy**, **65.06% macro F1**, **95.48% binary F1**, and **99.97% binary attack recall**, outperforming Random Forest (95.33% binary F1) and MLP (92.62% binary F1) across all ten individual attack categories. PyTorch MLP achieves maximum throughput exceeding **3.05 million flows/second** during CPU inference (2.83M flows/s on CICIDS episode test).

On the **corrected CICIDS episode-aware split** (where all 15 attack classes are present across train, validation, and test—superseding the obsolete chronological split where `Bot` and `Infilteration` were absent from training), 100K-budget balanced baselines achieve test **Macro F1 of 0.5590 (RF), 0.5093 (XGBoost), and 0.4531 (MLP)** with **Binary Average Precision of 0.8578–0.9449** and **Binary Recall of 91.45%–97.93%**, compared to **0.049–0.066** Macro F1 on the obsolete chronological split. However, full inverse-frequency class weighting induces high benign false-positive rates at the 0.5 threshold (`Binary F1 = 0.3170–0.4392` vs. `0.8675–0.9489` when unweighted on validation), lagging behind Vighnesh's full-train 3-seed **E-GraphSAGE** (`Macro F1 = 0.651`, `Binary F1 = 0.939`) and **E-GraphSAGE + Attention** (`Macro F1 = 0.672`, `Binary F1 = 0.931`) trained with milder `inverse_sqrt` class weighting. Bidirectional zero-shot cross-dataset transfer on the corrected episode split still degrades severely (`Test Binary F1 = 0.0025–0.1746`), matching E-GraphSAGE's transfer collapse (`0.158` UNSW->CICIDS, `0.020` CICIDS->UNSW) and confirming that cross-network domain shift is a genuine distribution gap rather than a split artifact.

---

## 2. Dataset and Feature Space

Both benchmark datasets share an identical 39-dimensional standardized NetFlow v2 feature space (`feature_names.joblib` verified 1:1 aligned across UNSW and CICIDS episode).

### 2.1 Benchmark Datasets
1. **NF-UNSW-NB15-v2** (`graph_data_bundle/data/graphs_unsw/`, preserved): 476 total graphs (2,380,000 flows) across 10 classes (*Analysis, Backdoor, Benign, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms*; `Benign` = index `2`).
   * **Train Split**: 334 graphs (1,670,000 flows)
   * **Validation Split**: 47 graphs (235,000 flows; 219,489 Benign vs. 15,511 Attack)
   * **Held-Out Test Split**: 95 graphs (475,000 flows; 440,165 Benign vs. 34,835 Attack)

2. **NF-CSE-CIC-IDS2018-v2 — Corrected Episode-Aware Split** (`data/graphs_cicids_episode/` & `data/features_cicids_episode/`): 3,896 total graphs (19,480,000 flows) across 15 classes (*Benign, Bot, Brute_Force_-Web, Brute_Force_-XSS, DDOS_attack-HOIC, DDOS_attack-LOIC-UDP, DDoS_attacks-LOIC-HTTP, DoS_attacks-GoldenEye, DoS_attacks-Hulk, DoS_attacks-SlowHTTPTest, DoS_attacks-Slowloris, FTP-BruteForce, Infilteration, SQL_Injection, SSH-Bruteforce*; `Benign` = index `0`).
   * **Train Split**: 2,659 graphs (13,295,000 flows; 88.98% Benign, 11.02% Attack; 15/15 classes present). Baselines use a 100,000-flow stratified sample budget (`retain_all_<=500_then_proportional`).
   * **Validation Split**: 342 graphs (1,710,000 flows; 87.75% Benign, 12.25% Attack; 15/15 classes present).
   * **Held-Out Test Split**: 895 graphs (4,475,000 flows; 90.64% Benign, 9.36% Attack; 15/15 classes present).
   * *Note*: Obsolete chronological-split CICIDS reports (2,728 / 389 / 779 graphs) are archived in `reports/part2/archive_chronological_split/`.

### 2.2 Feature Preprocessing and Inversion Provenance
Both datasets apply a $\log(1 + x)$ transform on 21 skewed packet/byte features followed by a training-only `StandardScaler`. For CICIDS episode, `data/features_cicids_episode/scaler.joblib` was refitted strictly on the new CICIDS training split (`n_samples_seen_ = 13,299,160`). For cross-dataset evaluation, target features are inverted via the target `StandardScaler` and re-standardized via the source `StandardScaler` without refitting on validation or test data.

---

## 3. Model Methodology & Hyperparameters

- **Random Forest (RF)**: `n_estimators=50`, `max_depth=15`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, `seed=42` (plus 3-seed stability check `42, 43, 44`).
- **XGBoost**: `n_estimators=50`, `max_depth=6`, `learning_rate=0.1`, `tree_method="hist"`, `LabelEncoder` projection via `XGBModelWrapper`, `balanced` sample weights, `seed=42`.
- **PyTorch MLP**: `39 -> 64 -> 32 -> C` (`ReLU`, `dropout=0.2`), Adam (`lr=0.001`), `batch_size=512`, `max_epochs=10`, early stopping `patience=3` on validation loss, full-space balanced `CrossEntropyLoss`, `seed=42`.

---

## 4. Empirical Results: Validation vs. Held-Out Test Evaluation

### 4.1 NF-UNSW-NB15-v2 Benchmark (Preserved: Val 235,000 flows | Test 475,000 flows)

| Model | Split | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Binary F1 | Binary AP | Inference Speed |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | Validation | 0.9759 | 0.6047 | 0.9793 | 0.9013 | 0.9995 | 0.9478 | 0.9752 | — |
| **Random Forest** | **Held-Out Test** | 0.9726 | 0.6136 | 0.9765 | 0.9115 | 0.9991 | 0.9533 | **0.9783** | 658,882 flows/s |
| **XGBoost** | Validation | 0.9784 | 0.6516 | 0.9819 | 0.9079 | 0.9995 | 0.9515 | 0.9506 | — |
| **XGBoost** | **Held-Out Test** | **0.9747** | **0.6506** | **0.9787** | **0.9137** | **0.9997** | **0.9548** | 0.9498 | 97,435 flows/s |
| **PyTorch MLP** | Validation | 0.9671 | 0.5079 | 0.9725 | 0.8356 | 0.9985 | 0.9098 | 0.9265 | — |
| **PyTorch MLP** | **Held-Out Test** | 0.9650 | 0.5237 | 0.9704 | 0.8636 | 0.9986 | 0.9262 | 0.9356 | **3,055,633 flows/s** |

### 4.2 NF-CSE-CIC-IDS2018-v2 Corrected Episode Split (100K Train Budget | Val 1,710,000 | Test 4,475,000 flows)

| Model | Split | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Binary F1 | Binary AP | Inference Speed |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | Validation | 0.7546 | 0.5994 | 0.8452 | 0.3395 | 0.9432 | 0.4992 | 0.8785 | — |
| **Random Forest** | **Held-Out Test** | **0.7720** | **0.5590** | **0.8591** | **0.2890** | 0.9145 | **0.4392** | 0.8578 | 161,071 flows/s |
| **XGBoost** | Validation | 0.7230 | 0.5279 | 0.8251 | 0.3135 | 0.9633 | 0.4730 | **0.9595** | — |
| **XGBoost** | **Held-Out Test** | 0.7432 | 0.5093 | 0.8419 | 0.2683 | 0.9572 | 0.4191 | 0.9435 | 146,439 flows/s |
| **PyTorch MLP** | Validation | 0.5711 | 0.5048 | 0.7032 | 0.2248 | **0.9799** | 0.3657 | 0.9188 | — |
| **PyTorch MLP** | **Held-Out Test** | 0.5966 | 0.4531 | 0.7280 | 0.1891 | **0.9793** | 0.3170 | **0.9449** | **2,828,077 flows/s** |

Full LaTeX and per-class tables are exported in `reports/part2/tables/val_vs_test_cicids.tex` and `reports/part2/tables/cicids_test_per_class.{csv,md,tex}`.

---

## 5. Class Imbalance Handling & Multi-Seed Stability

### 5.1 UNSW Class Weighting Ablation (Preserved, 50K Budget)
| Model | Weighting Policy | Accuracy | Macro F1 | Precision | Recall | Binary F1 | Binary AP | Attack Miss Rate |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | `balanced_subsample` | 0.9596 | **0.4774** | 0.8105 | **0.9279** | **0.8652** | 0.9035 | **7.21%** |
| **Random Forest** | `none` | 0.9581 | 0.4189 | 0.9691 | 0.4791 | 0.6412 | 0.9030 | 52.09% |
| **XGBoost** | `balanced` | 0.9624 | **0.4793** | 0.8263 | **0.9985** | **0.9043** | 0.8998 | **0.15%** |
| **XGBoost** | `none` | 0.9567 | 0.4610 | 0.9589 | 0.4214 | 0.5855 | 0.9276 | 57.86% |
| **PyTorch MLP** | `balanced` | 0.9286 | **0.3361** | 0.6394 | **0.9593** | **0.7674** | 0.7062 | **4.07%** |
| **PyTorch MLP** | `none` | 0.9356 | 0.1245 | 0.8376 | 0.2724 | 0.4111 | 0.7981 | 72.76% |

### 5.2 CICIDS Episode Class Weighting Ablation (100K Budget, Seed 42, Validation)
| Model | Weighting Policy | Accuracy | Macro F1 | Precision | Recall | Binary F1 | Binary AP | Attack Miss Rate |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | `balanced_subsample` | 0.7546 | **0.5994** | 0.3395 | **0.9432** | 0.4992 | 0.8785 | **5.68%** |
| **Random Forest** | `none` | **0.9723** | 0.5905 | **0.9892** | 0.8954 | **0.9400** | **0.9647** | 10.46% |
| **XGBoost** | `balanced` | 0.7230 | **0.5279** | 0.3135 | **0.9633** | 0.4730 | **0.9595** | **3.67%** |
| **XGBoost** | `none` | **0.9542** | 0.4280 | **0.9712** | 0.7838 | **0.8675** | 0.9549 | 21.62% |
| **PyTorch MLP** | `balanced` | 0.5711 | **0.5048** | 0.2248 | **0.9799** | 0.3657 | 0.9188 | **2.01%** |
| **PyTorch MLP** | `none` | **0.9813** | 0.4922 | **0.9746** | 0.9244 | **0.9489** | **0.9304** | 7.56% |

### 5.3 Multi-Seed Stability (Random Forest, Seeds 42, 43, 44)
- **UNSW (50K, preserved)**: Accuracy `0.9601 ± 0.0016`, Macro F1 `0.4794 ± 0.0107`, Binary F1 `0.8651 ± 0.0108`.
- **CICIDS Episode (100K)**: Accuracy `0.7515 ± 0.0097`, Macro F1 `0.5892 ± 0.0135`, Binary Precision `0.3357 ± 0.0088`, Binary Recall `0.9365 ± 0.0094`, Binary F1 `0.4942 ± 0.0090`, Binary AP `0.8693 ± 0.0122`.
- XGBoost and MLP remain single-seed (`seed=42`) per the established Part 2 compute allocation.

---

## 6. Cross-Dataset Generalization: UNSW <-> CICIDS (Corrected Episode Split)

| Transfer Setting | Split | Model | Target Flows | Accuracy | Binary Precision | Binary Recall | Binary F1 | Binary AP |
|---|---|---|---:|---|---|---|---|---|
| **UNSW -> CICIDS (Episode)** | Val | Random Forest | 1,710,000 | 0.6016 | 0.0104 | 0.0239 | 0.0145 | 0.1144 |
| **UNSW -> CICIDS (Episode)** | Val | XGBoost | 1,710,000 | 0.2820 | 0.1244 | 0.8052 | 0.2155 | 0.1135 |
| **UNSW -> CICIDS (Episode)** | Val | PyTorch MLP | 1,710,000 | 0.6677 | 0.0870 | 0.1804 | 0.1174 | 0.1043 |
| **UNSW -> CICIDS (Episode)** | **Test** | Random Forest | 4,475,000 | 0.6295 | 0.0118 | 0.0358 | 0.0177 | 0.0910 |
| **UNSW -> CICIDS (Episode)** | **Test** | **XGBoost** | 4,475,000 | 0.2803 | **0.0978** | **0.8136** | **0.1746** | **0.0976** |
| **UNSW -> CICIDS (Episode)** | **Test** | PyTorch MLP | 4,475,000 | **0.6669** | 0.0619 | 0.1806 | 0.0922 | 0.0788 |
| **CICIDS (Episode) -> UNSW** | Val | Random Forest | 235,000 | 0.9144 | 0.0064 | 0.0019 | 0.0030 | 0.0371 |
| **CICIDS (Episode) -> UNSW** | Val | XGBoost | 235,000 | 0.6251 | 0.0658 | 0.3544 | 0.1110 | 0.0619 |
| **CICIDS (Episode) -> UNSW** | Val | PyTorch MLP | 235,000 | 0.3218 | 0.0734 | 0.7976 | 0.1344 | 0.0595 |
| **CICIDS (Episode) -> UNSW** | **Test** | Random Forest | 475,000 | **0.9070** | 0.0059 | 0.0016 | 0.0025 | 0.0412 |
| **CICIDS (Episode) -> UNSW** | **Test** | XGBoost | 475,000 | 0.6099 | 0.0672 | 0.3352 | 0.1119 | 0.0640 |
| **CICIDS (Episode) -> UNSW** | **Test** | **PyTorch MLP** | 475,000 | 0.3385 | **0.0849** | **0.8195** | **0.1538** | **0.0676** |

---

## 7. Comparison Against Teammate E-GraphSAGE (Corrected CICIDS Episode Split)

| Model | Train Budget | Seeds | Test Accuracy | Test Macro F1 | Test Binary F1 | UNSW->CICIDS Test Bin F1 | CICIDS->UNSW Test Bin F1 |
|---|---|---|---|---|---|---|---|
| **Random Forest** (`balanced_subsample`) | 100,000 | 1 (`42`) | 0.7720 | 0.5590 | 0.4392 | 0.0177 | 0.0025 |
| **XGBoost** (`balanced`) | 100,000 | 1 (`42`) | 0.7432 | 0.5093 | 0.4191 | **0.1746** | 0.1119 |
| **PyTorch MLP** (`balanced`) | 100,000 | 1 (`42`) | 0.5966 | 0.4531 | 0.3170 | 0.0922 | **0.1538** |
| **E-GraphSAGE (mean, `inverse_sqrt`)** | 13,295,000 (Full) | 3 (`0,1,2`) | **0.9825** | 0.651 | **0.939** | 0.158 ± 0.080 | 0.020 ± 0.014 |
| **E-GraphSAGE + Attention (`inverse_sqrt`)** | 13,295,000 (Full) | 3 (`0,1,2`) | 0.9813 | **0.672** | 0.931 | — | — |

Key comparison notes:
1. E-GraphSAGE models were trained on the full 13.295M-flow training split with `inverse_sqrt` class weights across 3 seeds, whereas baselines used the 100K CPU budget (`seed=42`) and full `balanced` weighting.
2. In-domain on CICIDS episode test, E-GraphSAGE+Attention achieves substantially higher Macro F1 (`0.672` vs. `0.5590` RF) and Binary F1 (`0.931–0.939` vs. `0.3170–0.4392` weighted baselines; note unweighted RF/MLP reach `0.9400`/`0.9489` Binary F1 on validation).
3. Under zero-shot cross-dataset transfer (`test` split), **both** tabular baselines and E-GraphSAGE collapse (`Binary F1 <= 0.175`), demonstrating severe cross-domain shift even with episode-aware splitting and full 15-class training coverage.
