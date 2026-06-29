"""Causal intervention layer for sonde.

R-1 scope: additive steering over an nnterp ``StandardizedTransformer``, driven
either programmatically or from ``SteeringParams`` YAML. Directional ablation,
``TokenSelector``-driven per-row positions, activation patching, and per-head
steering land in R-2 / R-5 / R-6 — see ``docs/intervention_design.md``.
"""

from .context import InterventionContext
from .types import PendingSteer
from .vectors import load_vector

__all__ = [
    "InterventionContext",
    "PendingSteer",
    "load_vector",
]
