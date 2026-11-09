# Cross‑Dataset Baseline Evaluation (Stage 3B.2)

## Motivation

The goal of this stage is to assess how well a model trained on one intrusion‑
detection dataset (UNSW‑NB15 or CIC‑IDS‑2017) transfers to the other dataset.  This
helps understand the generalisation capability of classic machine‑learning
baselines – Random Forest, XGBoost and a shallow PyTorch MLP – when the feature
distribution changes.

## Datasets & Feature Compatibility

Both datasets are pre‑processed to a common *39‑dimensional* feature space.  The
pre‑processing pipelines (`src/preprocess_features.py` for UNSW and
`src/preprocess_cicids.py` for CICIDS) perform the same operations:

1. Drop identifier columns and the original `Label` column.
2. Apply `np.log1p` to a fixed list of *log columns* (see
   `src/eval/preprocess_cross.py`).
3. Standardise with a `StandardScaler` that is saved to
   `graph_data_bundle/data/features*/scaler.joblib`.
4. Store the column ordering in `feature_names.joblib`.

The cross‑dataset evaluator validates that the source model's required features
are present in the target set and that the log‑column semantics match.  Missing
or extra features trigger a clear failure rather than silent masking.

## Model Checkpoints

Training scripts now emit a checkpoint that can be re‑used for inference:

* **RandomForest** – `model.joblib` (Joblib‑dumped `RandomForestClassifier`).
* **XGBoost** – `model.joblib` (Joblib‑dumped `XGBClassifier`).
* **MLP** – `model.pt` (PyTorch `state_dict`).

These files are written alongside the usual artefacts in the `results/*`
directory.

## Cross‑Dataset Evaluation CLI

```
python -m src.eval.cross_dataset \
    --model rf \
    --source-dataset unsw \
    --target-dataset cicids \
    --checkpoint results/rf_unsw/model.joblib \
    --split val \
    --output-dir cross_unsw_to_cicids
```

The script performs the following steps:

1. **Load source artefacts** – scaler, feature names, class names and the model
   checkpoint.
2. **Load target split** – using `DataLoader.load_split_with_ids` to retain
   graph and edge identifiers.
3. **Feature transformation** – inverse‑transform the target scaler, reorder
   columns, revert the `log1p` operation (`np.expm1`), and finally forward‑
   transform with the source scaler.
4. **Inference** – obtain multiclass predictions and probabilities, then derive
   binary attack scores using the source benign class index.
5. **Metrics** – binary accuracy, precision, recall, F1, specificity, false
   positive/negative rates, and average‑precision.  Results are written to
   `metrics.json`/`metrics.csv`.
6. **Export** – predictions (including identifiers) are saved as `predictions.npz`
   using the shared utility.
7. **Run configuration** – a lightweight `run_config.json` records provenance.

## Binary Mapping Policy

The source model remains multiclass.  Binary labels are derived as:

```
binary = (multiclass_label != source_benign_id)
```

Probability for the *attack* class is `1 - P(Benign)`, where `P(Benign)` is the
probability column corresponding to the source benign class.

## Validation & Test Isolation

The default split is `val`.  Evaluating the `test` split requires the explicit
`--confirm-test` flag to avoid accidental leakage.

## Current Status

| Direction          | IMPLEMENTED | SYNTHETIC‑TESTED | REAL‑DATA VERIFIED |
|--------------------|-------------|------------------|---------------------|
| UNSW → CICIDS      | ✅           | ✅                | ✅ (using generated checkpoints) |
| CICIDS → UNSW      | ✅           | ✅                | ✅ (using generated checkpoints) |

No blockers remain for Stage 3B.2. The next step (Stage 4) will extend the
analysis to Graph Neural Network baselines.
