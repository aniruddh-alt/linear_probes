from __future__ import annotations

import torch
from torch import nn

from .base import BaseProbe


class MeanProbe(BaseProbe):
    """Mask-aware mean pooling then linear readout.

    6-stage decomposition:
        transform: identity
        score:     per-position linear readout (B, S, num_classes)
        aggregate: masked mean over the sequence dimension

    Mathematically equivalent to ``linear(masked_mean(h))`` because the readout
    is affine; we apply it per position so that subclasses can swap pooling
    without re-deriving the algebra.
    """

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.linear = nn.Linear(input_dim, num_classes)

    def score(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

    def aggregate(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return scores.mean(dim=1)
        m = mask.unsqueeze(-1).to(scores.dtype)
        return (scores * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
