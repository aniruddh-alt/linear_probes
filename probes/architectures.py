"""Probe architectures with a common (batch, seq, dim) -> (batch, 1) interface."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LinearProbe(nn.Module):
    """Last-token linear probe. Takes last token from sequence, applies linear."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.linear(x)


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


class MaxProbe(nn.Module):
    """Element-wise max pooling then linear."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            return self.linear(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        pooled = x.max(dim=1).values
        return self.linear(pooled)


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


class AttentionProbe(nn.Module):
    """Learned single-head attention pooling then linear.

    f(A) = w * (sum softmax(q*a_i) * V*a_i) + b
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, 1, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            return self.output(self.value(x))
        attn_scores = self.query(x).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        values = self.value(x)
        pooled = (attn_weights * values).sum(dim=1)
        return self.output(pooled)


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
        B, S, D = x.shape
        T = min(self.window_size, S)
        pooled = torch.nn.functional.avg_pool1d(
            x.transpose(1, 2), kernel_size=T, stride=1
        )
        max_pooled = pooled.max(dim=2).values
        return self.linear(max_pooled)


_PROBE_REGISTRY: dict[str, type[nn.Module]] = {
    "linear": LinearProbe,
    "mean": MeanProbe,
    "max": MaxProbe,
    "softmax": SoftmaxProbe,
    "attention": AttentionProbe,
    "max_rolling_mean": MaxRollingMeanProbe,
}


def build_probe(probe_type: str, input_dim: int, **kwargs: Any) -> nn.Module:
    """Build a probe by type name."""
    cls = _PROBE_REGISTRY.get(probe_type)
    if cls is None:
        raise ValueError(
            f"Unknown probe_type '{probe_type}'. "
            f"Available: {sorted(_PROBE_REGISTRY.keys())}"
        )
    return cls(input_dim=input_dim, **kwargs)
