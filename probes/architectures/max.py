from __future__ import annotations

import torch
from torch import nn


class MaxProbe(nn.Module):
    """Element-wise max pooling then linear."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            return self.linear(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        pooled = x.max(dim=1).values
        return self.linear(pooled)
