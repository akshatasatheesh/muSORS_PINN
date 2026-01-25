"""A simple vanilla MLP for Raman spectra -> concentrations."""

from __future__ import annotations

import torch
import torch.nn as nn


class VanillaMLP(nn.Module):
    """Basic fully-connected network.

    Input: spectrum vector (D)
    Output: 3 targets (order must match sample_submission.csv)

    Notes
    -----
    Targets are non-negative in practice, but if you enable target standardization
    (which can produce negative values), you should disable `non_negative`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(512, 256, 128),
        dropout: float = 0.15,
        non_negative: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
        self.non_negative = bool(non_negative)
        self.out_act = nn.Softplus() if self.non_negative else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.out_act(y)
