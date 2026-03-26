"""Probe architectures with a common (batch, seq, dim) -> (batch, 1) interface."""

from __future__ import annotations

from typing import Any

from torch import nn

from .attention import AttentionProbe
from .linear import LinearProbe
from .max import MaxProbe
from .max_rolling_mean import MaxRollingMeanProbe
from .mean import MeanProbe
from .softmax import SoftmaxProbe

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


__all__ = [
    "AttentionProbe",
    "LinearProbe",
    "MaxProbe",
    "MaxRollingMeanProbe",
    "MeanProbe",
    "SoftmaxProbe",
    "build_probe",
]
