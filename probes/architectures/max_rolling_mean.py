from __future__ import annotations

import torch
from torch import nn


class MaxRollingMeanProbe(nn.Module):
    """Rolling window mean -> element-wise max -> linear.

    f(A) = w * max_j(mean(a_{j:j+T})) + b
    """

    def __init__(self, input_dim: int, window_size: int = 40):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            return self.linear(x)
        _B, S, _D = x.shape
        T = min(self.window_size, S)
        pooled = torch.nn.functional.avg_pool1d(
            x.transpose(1, 2), kernel_size=T, stride=1
        )
        max_pooled = pooled.max(dim=2).values
        return self.linear(max_pooled)
