from __future__ import annotations

import einops
import torch
from torch import Tensor, nn

from .base import BaseProbe


class AttentionProbe(BaseProbe):
    """Multi-head attention pooling probe.

    6-stage decomposition:
        transform: identity
        score:     per-head attention logits ``(h · W_q) / sqrt(D)``  -> (B, S, H)
        aggregate: softmax over positions per head, weighted sum of ``h``,
                   flatten heads, linear readout
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        num_classes: int = 1,
    ):
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.n_heads = n_heads
        self.scale = input_dim**0.5
        self.W_q = nn.Parameter(torch.empty(input_dim, n_heads))
        self.W_out = nn.Parameter(torch.empty(n_heads * input_dim, num_classes))
        self.b_out = nn.Parameter(torch.zeros(num_classes))

        nn.init.normal_(self.W_q, std=input_dim**-0.5)
        nn.init.normal_(self.W_out, std=(n_heads * input_dim) ** -0.5)

        self.attention_weights_: Tensor | None = None

    def score(self, h: Tensor) -> Tensor:
        return einops.einsum(h, self.W_q, "b s d, d n -> b s n") / self.scale

    def aggregate(
        self,
        scores: Tensor,
        h: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
        attention_weights = scores.softmax(dim=1)
        self.attention_weights_ = attention_weights.detach()

        context = einops.einsum(attention_weights, h, "b s n, b s d -> b n d")
        context = context.flatten(start_dim=1)
        return einops.einsum(context, self.W_out, "b h, h c -> b c") + self.b_out

    @property
    def direction(self) -> Tensor | None:
        """Multi-head attention probes don't have a single causal direction."""
        return None

    @property
    def bias(self) -> float | None:
        if self.b_out.numel() != 1:
            return None
        return float(self.b_out.detach().cpu().item())
