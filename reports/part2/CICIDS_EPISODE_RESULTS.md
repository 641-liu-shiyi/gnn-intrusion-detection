# Part 2 Results Summary — Corrected CICIDS-2018 Episode-Aware Split

**Dataset:** NF-CSE-CIC-IDS2018-v2 (Shiyi's episode-aware split: `data/graphs_cicids_episode/`, `data/features_cicids_episode/`)  
**Partitions:** 2,659 train graphs (13,295,000 flows) / 342 val graphs (1,710,000 flows) / 895 test graphs (4,475,000 flows)  
**Training Budget:** 100,000 stratified training flows (`retain_all_<=500_then_proportional`), single primary seed (`seed=42`); RF stability verified across 3 seeds (`42, 43, 44`).  
**Old Chronological Split:** Archived under `reports/part2/archive_chronological_split/` (superseded).

---

## 1. Main Baselines (Validation vs. Held-Out Test, 100K Budget, Seed 42)

| Model | Split | Accuracy | Macro F1 | Weighted F1 | Binary Prec | Binary Rec | Binary F1 | Binary AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** (`balanced_subsample`) | Validation | 0.7546 | **0.5994** | 0.8452 | 0.3395 | 0.9432 | **0.4992** | 0.8785 |
| **Random Forest** (`balanced_subsample`) | **Held-Out Test** | **0.7720** | **0.5590** | **0.8591** | **0.2890** | 0.9145 | **0.4392** | 0.8578 |
| **XGBoost** (`balanced`) | Validation | 0.7230 | 0.5279 | 0.8251 | 0.3135 | 0.9633 | 0.4730 | **0.9595** |
| **XGBoost** (`balanced`) | **Held-Out Test** | 0.7432 | 0.5093 | 0.8419 | 0.2683 | 0.9572 | 0.4191 | 0.9435 |
| **PyTorch MLP** (`balanced`) | Validation | 0.5711 | 0.5048 | 0.7032 | 0.2248 | **0.9799** | 0.3657 | 0.9188 |
| **PyTorch MLP** (`balanced`) | **Held-Out Test** | 0.5966 | 0.4531 | 0.7280 | 0.1891 | **0.9793** | 0.3170 | **0.9449** |

Held-out test binary confusion counts (4,475,000 flows = 4,056,177 Benign + 418,823 Attack):
- **Random Forest**: TN = 3,113,883 | FP = 942,294 | FN = 35,790 | TP = 383,033
- **XGBoost**: TN = 2,962,833 | FP = 1,093,344 | FN = 17,925 | TP = 400,898
- **PyTorch MLP**: TN = 2,297,190 | FP = 1,758,987 | FN = 8,669 | TP = 410,154

---

## 2. Class-Weighting Ablation (Validation, 100K Budget, Seed 42)

| Model | Policy | Accuracy | Macro F1 | Binary Prec | Binary Rec | Binary F1 | Binary AP | Miss Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | `balanced_subsample` | 0.7546 | **0.5994** | 0.3395 | **0.9432** | 0.4992 | 0.8785 | **5.68%** |
| **Random Forest** | `none` | **0.9723** | 0.5905 | **0.9892** | 0.8954 | **0.9400** | **0.9647** | 10.46% |
| **XGBoost** | `balanced` | 0.7230 | **0.5279** | 0.3135 | **0.9633** | 0.4730 | **0.9595** | **3.67%** |
| **XGBoost** | `none` | **0.9542** | 0.4280 | **0.9712** | 0.7838 | **0.8675** | 0.9549 | 21.62% |
| **PyTorch MLP** | `balanced` | 0.5711 | **0.5048** | 0.2248 | **0.9799** | 0.3657 | 0.9188 | **2.01%** |
| **PyTorch MLP** | `none` | **0.9813** | 0.4922 | **0.9746** | 0.9244 | **0.9489** | **0.9304** | 7.56% |

**Key Takeaway:** On the 15-class CICIDS episode split, full inverse-frequency weighting (`balanced` / `balanced_subsample`) maximizes attack recall (94.3%–98.0%) and rare-class Macro F1 (`+9.99 pp` on XGBoost), but introduces substantial benign false positives at the 0.5 threshold (`Binary Precision` 0.225–0.339). Unweighted training (`none`) yields high Binary F1 (0.868–0.949) and accuracy (95.4%–98.1%) at the cost of higher missed attacks (21.6% miss rate on XGBoost) and lower Macro F1.

### Multi-Seed Stability (Random Forest, 100K Budget, Seeds 42, 43, 44, Validation)
- **Accuracy**: `0.7515 ± 0.0097`
- **Macro F1**: `0.5892 ± 0.0135`
- **Weighted F1**: `0.8428 ± 0.0063`
- **Binary Precision**: `0.3357 ± 0.0088`
- **Binary Recall**: `0.9365 ± 0.0094`
- **Binary F1**: `0.4942 ± 0.0090`
- **Binary AP**: `0.8693 ± 0.0122`

---

## 3. Cross-Dataset Transfer (Corrected Episode Scalers & Splits)

| Direction | Split | Model | Target Flows | Accuracy | Binary Prec | Binary Rec | Binary F1 | Binary AP |
|---|---|---|---:|---:|---:|---:|---:|---:|
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

## 4. Comparison Against E-GraphSAGE (Corrected CICIDS Episode Test Split)

Reference GNN values from Member 3 (`member3/results_cicids_episode/README.md` & `summary_3seeds.txt`):

| Model | Train Budget | Seeds | Weighting | Test Accuracy | Test Macro F1 | Test Binary F1 | UNSW->CICIDS Test Bin F1 | CICIDS->UNSW Test Bin F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| **Random Forest** | 100,000 | 1 (`42`) | `balanced_subsample` | 0.7720 | 0.5590 | 0.4392 | 0.0177 | 0.0025 |
| **XGBoost** | 100,000 | 1 (`42`) | `balanced` | 0.7432 | 0.5093 | 0.4191 | **0.1746** | 0.1119 |
| **PyTorch MLP** | 100,000 | 1 (`42`) | `balanced` | 0.5966 | 0.4531 | 0.3170 | 0.0922 | **0.1538** |
| **E-GraphSAGE (mean)** | 13,295,000 (Full) | 3 (`0,1,2`) | `inverse_sqrt` | **0.9825** | 0.651 | **0.939** | 0.158 ± 0.080 | 0.020 ± 0.014 |
| **E-GraphSAGE + Attention** | 13,295,000 (Full) | 3 (`0,1,2`) | `inverse_sqrt` | 0.9813 | **0.672** | 0.931 | — | — |

**Protocol Differences to Note:**
1. **Training Budget**: Baselines trained on `100,000` sampled training flows (`0.75%` of the 13.295M training split) on CPU; E-GraphSAGE trained on all `13,295,000` flows across 2,659 graphs on GPU.
2. **Seeds**: Main CICIDS baseline test evaluations use a single primary seed (`seed=42`; RF 3-seed validation stability measured separately at `0.5892 ± 0.0135` Macro F1), whereas E-GraphSAGE metrics are 3-seed means (`0, 1, 2`).
3. **Class Weighting**: Baselines use full `balanced` / `balanced_subsample` weights, whereas E-GraphSAGE uses milder `inverse_sqrt` weights (`0.931–0.939` Binary F1; unweighted RF/MLP reach `0.9400` / `0.9489` Binary F1 on validation).
4. **Cross-Dataset Transfer**: Both tabular baselines and E-GraphSAGE suffer severe zero-shot degradation (`Binary F1 <= 0.175` on `test`) across both directions, confirming that cross-network domain shift affects both flow-level and graph-level classifiers.
