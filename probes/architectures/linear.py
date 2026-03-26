from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    """Last-token linear probe. Takes last token from sequence, applies linear."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.linear(x)
