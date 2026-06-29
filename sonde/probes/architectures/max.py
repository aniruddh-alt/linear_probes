from __future__ import annotations

import torch
from torch import nn

from .base import BaseProbe


class MaxProbe(BaseProbe):
    """Element-wise max pooling then linear readout.

    6-stage decomposition:
        transform: identity
        score:     pass-through (``max`` does not commute with ``linear``, so
                   we score after pooling)
        aggregate: masked element-wise max over the sequence dimension,
                   followed by the linear readout
    """

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.linear = nn.Linear(input_dim, num_classes)

    def score(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def aggregate(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        pooled = h.max(dim=1).values
        return self.linear(pooled)
