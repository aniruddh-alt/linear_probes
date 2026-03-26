from __future__ import annotations

import einops
import torch
from torch import Tensor, nn


class AttentionProbe(nn.Module):
    """Multi-head attention pooling probe."""

    def __init__(self, input_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.scale = input_dim**0.5
        self.W_q = nn.Parameter(torch.empty(input_dim, n_heads))
        self.W_out = nn.Parameter(torch.empty(n_heads * input_dim, 1))
        self.b_out = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.W_q, std=input_dim**-0.5)
        nn.init.normal_(self.W_out, std=(n_heads * input_dim) ** -0.5)

        self.attention_weights_: Tensor | None = None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        attention_logits = (
            einops.einsum(x, self.W_q, "b s d, d n -> b s n") / self.scale
        )

        if mask is not None:
            attention_logits = attention_logits.masked_fill(
                ~mask.unsqueeze(-1).bool(), float("-inf")
            )

        attention_weights = attention_logits.softmax(dim=1)
        self.attention_weights_ = attention_weights.detach()

        context = einops.einsum(attention_weights, x, "b s n, b s d -> b n d")
        context = context.flatten(start_dim=1)

        logits = (
            einops.einsum(context, self.W_out, "b h, h one -> b one")
            + self.b_out
        )

        return logits
