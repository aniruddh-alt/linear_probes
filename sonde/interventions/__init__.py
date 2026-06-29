"""Causal intervention layer for sonde.

Additive steering and directional ablation (``project_subtract``) over an
nnterp ``StandardizedTransformer``, driven programmatically or from
``SteeringParams`` YAML, applied uniformly inside ``trace`` and across
``generate`` decode steps. Per-row ``TokenSelector`` position masks, activation
patching, and per-head steering are future work — see
``docs/intervention_design.md``.
"""

from .context import InterventionContext
from .types import PendingSteer
from .vectors import load_vector

__all__ = [
    "InterventionContext",
    "PendingSteer",
    "load_vector",
]
