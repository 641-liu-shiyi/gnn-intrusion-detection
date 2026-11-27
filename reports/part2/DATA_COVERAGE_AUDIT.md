# CS5814 Part 2: Final Data Coverage and Experiment Audit Report (UNSW + Corrected CICIDS Episode Split)

**Repository Branch:** `feature/cicids-episode-baselines`  
**Audit Script Provenance:** `scripts/audit_physical_data.py` & `scripts/audit_full_experiments.py`  
**Authoritative Evidence Sources:**
- Physical graph & feature directories: `graph_data_bundle/data/{graphs_unsw,features}/` (UNSW) and `data/{graphs_cicids_episode,features_cicids_episode}/` (CICIDS episode split)
- Saved `run_config.json`, `metrics.json`, checkpoints (`model.joblib`, `model.pt`), and flow-level `predictions.npz` archives.
- Archived chronological CICIDS audit: `reports/part2/archive_chronological_split/DATA_COVERAGE_AUDIT_CHRONOLOGICAL.md`

---

## 1. Physical Dataset Summary

| Dataset | Train Graphs (Flows) | Val Graphs (Flows) | Test Graphs (Flows) | Total Flows | Classes Present (Train/Val/Test) | Benign ID |
|---|---:|---:|---:|---:|---:|---:|
| **UNSW-NB15** (preserved) | 334 (1,670,000) | 47 (235,000) | 95 (475,000) | 2,380,000 | 10 / 10 / 10 | `2` |
| **CICIDS-2018 (Episode Split)** | 2,659 (13,295,000) | 342 (1,710,000) | 895 (4,475,000) | 19,480,000 | **15 / 15 / 15** | `0` |

*Key Fix Verified*: Unlike the obsolete chronological split (2,728 / 389 / 779 graphs, where `Bot` and `Infilteration` had 0 training flows), Shiyi's episode-aware split contains **all 15 classes** in `train`, `val`, and `test`, with 0 NaN/Inf values, 0 binary label mismatches, and a refitted 39-feature training-only `StandardScaler` (`n_samples_seen_ = 13,299,160`). See `reports/part2/CICIDS_EPISODE_DATA_AUDIT.md` for full per-class split counts.

---

## 2. Training Coverage & Stratified Sample Budget

- **UNSW-NB15 (preserved)**: 100.0% full training (`1,670,000` flows across 334 graphs, `max_samples=0`).
- **CICIDS-2018 Episode Split**: Bounded `100,000`-flow stratified training sample (`0.752%` of 13,295,000 training flows) using `retain_all_<=500_then_proportional`.
  - All 15 classes are sampled in training: `Brute_Force_-XSS` (336/336, 100%) and `SQL_Injection` (310/310, 100%) are retained in full; remaining 99,354 slots are allocated proportionally (`Benign`: 88,409, `DDOS_attack-HOIC`: 5,400, `DDoS_attacks-LOIC-HTTP`: 1,510, `FTP-BruteForce`: 1,012, `Infilteration`: 606, `Bot`: 577, `DoS_attacks-SlowHTTPTest`: 552, `DoS_attacks-Hulk`: 524, `SSH-Bruteforce`: 493, `DoS_attacks-GoldenEye`: 160, `DoS_attacks-Slowloris`: 94, `DDOS_attack-LOIC-UDP`: 9, `Brute_Force_-Web`: 8).

---

## 3. Evaluation & Cross-Dataset Coverage (100% of Target Splits)

| Scope | Partition | Flows Evaluated | Graphs | Models Verified |
|---|---|---:|---:|---|
| **UNSW Validation** | `val` | 235,000 | 47 | RF, XGBoost, MLP |
| **UNSW Held-Out Test** | `test` | 475,000 | 95 | RF, XGBoost, MLP |
| **CICIDS Episode Validation** | `val` | 1,710,000 | 342 | RF, XGBoost, MLP |
| **CICIDS Episode Held-Out Test** | `test` | 4,475,000 | 895 | RF, XGBoost, MLP |
| **UNSW -> CICIDS (Episode)** | `val` / `test` | 1,710,000 / 4,475,000 | 342 / 895 | RF, XGBoost, MLP |
| **CICIDS (Episode) -> UNSW** | `val` / `test` | 235,000 / 475,000 | 47 / 95 | RF, XGBoost, MLP |
