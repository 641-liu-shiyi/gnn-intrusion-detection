"""Simple feed-forward MLP for the UNSW intrusion-detection task.

The model follows the architecture that was agreed upon for the checkpoint:

```
input (39) -> Linear(39, 64) -> ReLU -> Dropout(0.2)
          -> Linear(64, 32) -> ReLU -> Dropout(0.2)
          -> Linear(32, 10)   # 10 classes in UNSW
```

Only a single ``nn.Module`` subclass is provided - the training script builds
the optimiser, loss function and data loaders.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MLP"]


class MLP(nn.Module):
    """Multilayer perceptron with two hidden layers and dropout.

    Parameters
    ----------
    input_dim : int, default 39
        Number of input features (the graph ``edge_attr`` column count).
    hidden1 : int, default 64
        Size of the first hidden layer.
    hidden2 : int, default 32
        Size of the second hidden layer.
    output_dim : int, default 10
        Number of output classes - UNSW defines 10 multiclass labels.
    dropout : float, default 0.2
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden1: int = 64,
        hidden2: int = 32,
        output_dim: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input feature matrix.

        Returns
        -------
        torch.Tensor of shape (batch_size, output_dim)
            Logits (raw, un-softmaxed scores) for each class.
        """
        return self.net(x)
