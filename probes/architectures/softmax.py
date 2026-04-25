from __future__ import annotations

import torch
from torch import nn

from .base import BaseProbe


class SoftmaxProbe(BaseProbe):
    """Learned temperature-weighted softmax pooling then linear readout.

        f(A) = w · ( Σ_i softmax(φ · w·a_i) · a_i ) + b

    6-stage decomposition:
        transform: identity
        score:     per-position attention logits ``φ · linear(h)``
        aggregate: softmax-weighted sum of ``h``, then the linear readout

    Currently binary-only; the per-position readout is reused as the attention
    score so the probe has a single direction in activation space. For
    multi-class we would need to split the attention readout from the class
    readout (TODO).
    """

    def __init__(self, input_dim: int, phi: float = 5.0, num_classes: int = 1):
        if num_classes != 1:
            raise NotImplementedError(
                "SoftmaxProbe currently supports binary (num_classes=1) only; "
                "the attention score is tied to the binary readout."
            )
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.linear = nn.Linear(input_dim, num_classes)
        self.phi = phi

    def score(self, h: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(h, self.linear.weight) * self.phi

    def aggregate(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_logits = scores.squeeze(-1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = (weights * h).sum(dim=1)
        return self.linear(pooled)
