from __future__ import annotations

import torch
from torch import nn

from .base import BaseProbe


class LinearProbe(BaseProbe):
    """Fixed-position linear probe (default: last token).

    6-stage decomposition:
        transform: identity
        score:     per-position linear readout (B, S, num_classes)
        aggregate: pick a single position (default ``-1``)
    """

    def __init__(
        self,
        input_dim: int,
        position: int = -1,
        num_classes: int = 1,
    ):
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.linear = nn.Linear(input_dim, num_classes)
        self.position = position

    def score(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

    def aggregate(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return scores[:, self.position, :]
