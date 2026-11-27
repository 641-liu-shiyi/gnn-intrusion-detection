| Direction | Split | Model | Target Flows | Accuracy | Binary Precision | Binary Recall | Binary F1 | Binary AP |
|---|---|---|---:|---|---|---|---|---|
| UNSW -> CICIDS (Episode) | val | Random Forest | 1,710,000 | 0.6016 | 0.0104 | 0.0239 | 0.0145 | 0.1144 |
| UNSW -> CICIDS (Episode) | val | XGBoost | 1,710,000 | 0.2820 | 0.1244 | 0.8052 | 0.2155 | 0.1135 |
| UNSW -> CICIDS (Episode) | val | PyTorch MLP | 1,710,000 | 0.6677 | 0.0870 | 0.1804 | 0.1174 | 0.1043 |
| UNSW -> CICIDS (Episode) | test | Random Forest | 4,475,000 | 0.6295 | 0.0118 | 0.0358 | 0.0177 | 0.0910 |
| UNSW -> CICIDS (Episode) | test | XGBoost | 4,475,000 | 0.2803 | 0.0978 | 0.8136 | 0.1746 | 0.0976 |
| UNSW -> CICIDS (Episode) | test | PyTorch MLP | 4,475,000 | 0.6669 | 0.0619 | 0.1806 | 0.0922 | 0.0788 |
| CICIDS (Episode) -> UNSW | val | Random Forest | 235,000 | 0.9144 | 0.0064 | 0.0019 | 0.0030 | 0.0371 |
| CICIDS (Episode) -> UNSW | val | XGBoost | 235,000 | 0.6251 | 0.0658 | 0.3544 | 0.1110 | 0.0619 |
| CICIDS (Episode) -> UNSW | val | PyTorch MLP | 235,000 | 0.3218 | 0.0734 | 0.7976 | 0.1344 | 0.0595 |
| CICIDS (Episode) -> UNSW | test | Random Forest | 475,000 | 0.9070 | 0.0059 | 0.0016 | 0.0025 | 0.0412 |
| CICIDS (Episode) -> UNSW | test | XGBoost | 475,000 | 0.6099 | 0.0672 | 0.3352 | 0.1119 | 0.0640 |
| CICIDS (Episode) -> UNSW | test | PyTorch MLP | 475,000 | 0.3385 | 0.0849 | 0.8195 | 0.1538 | 0.0676 |
