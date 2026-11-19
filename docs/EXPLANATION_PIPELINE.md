# LLM Alert Explanation Pipeline

Turns flows flagged by a detection model into short, grounded alerts. The LLM does not take part in detection.

## Data contract

| Input | Produced by | Fields used |
|---|---|---|
| `predictions.npz` | Any model exporting the shared schema in `src/comparison/gnn_importer.py` | `y_pred_multiclass`, `y_pred_binary`, `y_proba_multiclass`, `graph_id`, `edge_id`, `class_names`, `benign_class_id`, `dataset`, `split`; `y_true_multiclass` for alignment and review only |
| `{split}_graph_{id:05d}.npz` | `src/build_unsw_graphs.py`, `src/build_cicids_graphs.py` | `edge_index`, `edge_attr`, `edge_attack` |
| `scaler.joblib`, `feature_names.joblib` | `src/preprocess_features.py`, `src/preprocess_cicids.py` | Restore readable values from `edge_attr` |

`graph_id` is the index of the sorted graph file and `edge_id` is the row within that graph. The pipeline stops with an error if `y_true_multiclass` disagrees with `edge_attack` in any graph it reads.

## What the LLM receives

* **Model output:** predicted class, confidence, runner-up class and its probability.
* **Flow:** anonymized host labels, destination port, protocol, bytes and packets in each direction, duration, TCP flags.
* **Window context (5,000 flows):** source flow count, distinct destination hosts and ports, destination flow count and distinct sources, flows with the same source, destination host and port, and flagged flow counts.

Ground-truth labels, IP addresses and alert IDs are never sent. Duration is sent as unavailable when it is 0 or a NetFlow timer wraparound value (4,290,000 ms or more). In the UNSW test split every flow is one of these two cases.

## Alert selection

For each predicted attack class, the run samples up to `--per-cell` flows from each outcome: correct class, wrong attack class, and false positive. This keeps rare classes and model mistakes in the review set.

## Evaluation

`grounding.py` flags numbers absent from the input, attack classes other than the predicted or runner-up class, speculative terms such as malware or attacker, time words the window cannot support, judgment words, spelled-out numbers, a wrong opening, and missing uncertainty below 60% confidence. These flags only direct attention. A person scores `manual_factually_consistent` and `manual_clear` (`y` or `n`) in `review_sheet.csv`.

Alerts describe evidence consistent with a prediction. They are not attributions of what the model actually used.

## Running

```bash
# 1. Build evidence and prompts without calling the LLM
python -m scripts.run_explanations \
  --predictions PATH/TO/predictions.npz \
  --graphs data/graphs_unsw \
  --scaler data/features/scaler.joblib \
  --feature-names data/features/feature_names.joblib \
  --out results/explanations/unsw_gnn --dry-run

# 2. Generate alerts (VT network or VPN may be required)
export ARC_API_KEY=sk-...
python -m scripts.run_explanations [same arguments, without --dry-run] --model MODEL_NAME

# 3. Tally the hand-scored sheet
python -m scripts.run_explanations --summarize-review results/explanations/unsw_gnn/review_sheet.csv
```

Responses are cached in `llm_cache.jsonl`, so repeating a run reuses earlier alerts and resumes after a failure.
