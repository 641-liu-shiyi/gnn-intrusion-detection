# CIC-IDS2018 results on the episode-aware split (Member 3, E-GraphSAGE)

Data: Shiyi's episode-aware 70/10/20 split (commit e3c6ff5): 2,659 / 342 / 895 graphs, 5,000-flow windows,
39 features, all 15 classes present in train, val and test (checked before training by
`member3/cicids_episode_runs.sh`). Old chronological-split results (macro F1 0.065) are superseded.

Reproduce: `bash member3/cicids_episode_runs.sh` (configs `cicids_episode_final.json`, `cicids_episode_attention.json`).
Metrics from `src/eval/metrics.compute_metrics`; mean ± std over seeds 0, 1, 2 (`summary_3seeds.txt`).

## In-domain (CIC-IDS2018 test, 4,475,000 flows)

| Model | Val macro F1 | Test macro F1 | Test binary F1 | Test accuracy | Params |
|---|---|---|---|---|---|
| Final settings (mean, flow features in classifier, inverse-sqrt weights) | 0.702 ± 0.005 | 0.651 ± 0.004 | 0.939 ± 0.014 | 0.983 | 80,984 |
| Same, attention aggregation | 0.728 ± 0.007 | **0.672 ± 0.011** | 0.931 ± 0.012 | 0.981 | 98,264 |

Attention gives +0.021 test macro F1 (+0.026 val), larger than on UNSW (+0.011), with no gain in binary F1.

## Per-class errors (final settings, seed 0, test)

| Class | Train flows | Test flows | Predicted as |
|---|---|---|---|
| DoS-SlowHTTPTest | 73,890 | 21,110 | FTP-BruteForce 86%, SlowHTTPTest 9% |
| Infilteration | 81,068 | 23,157 | Benign 51%, Infilteration 48% |
| SQL_Injection | 310 | 84 | Brute_Force-XSS 75%, Brute_Force-Web 14% |
| Brute_Force-Web | 1,117 | 327 | Brute_Force-Web 78%, Brute_Force-XSS 21% |
| GoldenEye | 21,456 | 6,128 | GoldenEye 83%, Benign 17% |

Near-perfect (F1 ≥ 0.97): Benign, Bot, DDoS-HOIC, DDoS-LOIC-HTTP, Slowloris, SSH-Bruteforce.
Full confusion matrix: `final_cicids_confusion_test.csv` (rows = true, cols = predicted).

## Zero-shot transfer, binary F1 (3 seeds)

| Direction | Val | Test | In-domain (source) |
|---|---|---|---|
| UNSW → CIC-IDS2018 | 0.161 ± 0.078 | 0.158 ± 0.080 | 0.94–0.95 |
| CIC-IDS2018 → UNSW | 0.020 ± 0.014 | 0.020 ± 0.014 | 0.94–0.95 |

Per seed (test): UNSW → CIC 0.157 / 0.257 / 0.061; CIC → UNSW 0.001 / 0.026 / 0.034.
With every attack type now in CIC-IDS2018 training, CIC → UNSW still fails, so the transfer gap is a
real domain shift, not a split artefact. Baseline transfer numbers on this split are pending (Member 2).

Prediction exports (shared format) for seed 0 are on the server only (`member3/predictions/cicids_episode_final_seed0_{val,test}`, 96 MB / 252 MB).
