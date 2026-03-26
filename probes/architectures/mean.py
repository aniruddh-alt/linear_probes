from __future__ import annotations

import torch
from torch import nn


class MeanProbe(nn.Module):
    """Mask-aware mean pooling then linear."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            return self.linear(x)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1)
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.linear(pooled)
