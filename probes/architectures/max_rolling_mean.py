from __future__ import annotations

import torch
from torch import nn

from .base import BaseProbe


class MaxRollingMeanProbe(BaseProbe):
    """Rolling-window mean -> element-wise max -> linear readout.

        f(A) = w · max_j( mean(a_{j:j+T}) ) + b

    6-stage decomposition:
        transform: identity (the rolling-mean is positional smoothing, not a
                   per-position transform — we keep it inside ``aggregate`` so
                   the windowed reduction stays atomic)
        score:     pass-through
        aggregate: rolling-mean window over the sequence dim, element-wise max
                   across windows, linear readout

    From Kramár & Engels et al. 2026.
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int = 40,
        num_classes: int = 1,
    ):
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.linear = nn.Linear(input_dim, num_classes)
        self.window_size = window_size

    def score(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def aggregate(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        _B, S, _D = h.shape
        T = min(self.window_size, S)
        rolling = torch.nn.functional.avg_pool1d(
            h.transpose(1, 2), kernel_size=T, stride=1
        )
        max_pooled = rolling.max(dim=2).values
        return self.linear(max_pooled)
