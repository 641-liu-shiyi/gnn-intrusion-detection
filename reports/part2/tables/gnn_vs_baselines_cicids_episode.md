| Model | Family | Train Budget | Seeds | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Binary F1 | Test Binary AP |
|---|---|---|---|---|---|---|---|---|
| Random Forest (balanced_subsample) | Tabular Baseline | 100,000 | 1 (seed 42) | 0.7720 | 0.5590 | 0.8591 | 0.4392 | 0.8578 |
| XGBoost (balanced) | Tabular Baseline | 100,000 | 1 (seed 42) | 0.7432 | 0.5093 | 0.8419 | 0.4191 | 0.9435 |
| PyTorch MLP (balanced) | Tabular Baseline | 100,000 | 1 (seed 42) | 0.5966 | 0.4531 | 0.7280 | 0.3170 | 0.9449 |
| E-GraphSAGE (mean, inverse_sqrt) | Graph Neural Network | 13,295,000 (Full) | 3 (0, 1, 2) | 0.9825 | 0.651 | 0.9833 | 0.939 | N/A |
| E-GraphSAGE + Attention (inverse_sqrt) | Graph Neural Network | 13,295,000 (Full) | 3 (0, 1, 2) | 0.9813 | 0.672 | 0.9825 | 0.931 | N/A |
