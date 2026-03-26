from __future__ import annotations

import torch
from torch import nn


class SoftmaxProbe(nn.Module):
    """Learned temperature-weighted softmax pooling then linear.

    f(A) = w * (sum softmax(phi * w*a_i) * a_i) + b
    """

    def __init__(self, input_dim: int, phi: float = 5.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.phi = phi

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            return self.linear(x)
        scores = torch.nn.functional.linear(x, self.linear.weight) * self.phi
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (weights * x).sum(dim=1)
        return self.linear(pooled)
