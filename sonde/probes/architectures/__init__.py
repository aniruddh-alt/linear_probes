"""Probe architectures.

Every probe inherits from :class:`BaseProbe` and implements the Kramár &
Engels et al. (2026) 6-stage framework — see ``base.py`` and
``docs/toolkit_audit.md`` §6.
"""

from __future__ import annotations

from typing import Any

from .attention import AttentionProbe
from .base import BaseProbe
from .linear import LinearProbe
from .max import MaxProbe
from .max_rolling_mean import MaxRollingMeanProbe
from .mean import MeanProbe
from .softmax import SoftmaxProbe

_PROBE_REGISTRY: dict[str, type[BaseProbe]] = {
    "linear": LinearProbe,
    "mean": MeanProbe,
    "max": MaxProbe,
    "softmax": SoftmaxProbe,
    "attention": AttentionProbe,
    "max_rolling_mean": MaxRollingMeanProbe,
}


def build_probe(probe_type: str, input_dim: int, **kwargs: Any) -> BaseProbe:
    """Build a probe by type name.

    Returns a :class:`BaseProbe` instance so callers can rely on the
    common ``forward(x, mask=...)`` contract plus the ``direction`` /
    ``bias`` properties.
    """
    cls = _PROBE_REGISTRY.get(probe_type)
    if cls is None:
        raise ValueError(
            f"Unknown probe_type '{probe_type}'. "
            f"Available: {sorted(_PROBE_REGISTRY.keys())}"
        )
    return cls(input_dim=input_dim, **kwargs)


__all__ = [
    "AttentionProbe",
    "BaseProbe",
    "LinearProbe",
    "MaxProbe",
    "MaxRollingMeanProbe",
    "MeanProbe",
    "SoftmaxProbe",
    "build_probe",
]
