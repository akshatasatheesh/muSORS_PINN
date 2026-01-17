"""A simple vanilla MLP for Raman spectra -> concentrations."""

from __future__ import annotations

import torch
import torch.nn as nn


class VanillaMLP(nn.Module):
    """Basic fully-connected network.

    Input: spectrum vector (D)
    Output: 3 targets (glucose, sodium acetate, magnesium sulfate)

    Note: For this competition, targets are non-negative in practice, but we don't hard-clip.
    You can enable `non_negative=True` to apply Softplus to outputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(512, 256, 128),
        dropout: float = 0.15,
        non_negative: bool = True,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
        self.non_negative = non_negative
        self.out_act = nn.Softplus() if non_negative else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.out_act(y)
