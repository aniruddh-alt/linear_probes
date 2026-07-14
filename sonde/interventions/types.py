"""Dataclasses representing pending interventions on a transformer model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SteeringMode = Literal["additive", "project_subtract"]


@dataclass
class PendingSteer:
    """A configured-but-not-yet-applied steering operation.

    Vectors are stored normalised (unit L2) when ``normalize=True`` was used at
    construction time. ``factor``'s meaning depends on ``mode``: for
    ``"additive"`` it is the signed steering strength; for ``"project_subtract"``
    it is the ablation fraction (``1.0`` = full directional ablation).

    ``positions`` mirrors ``nnterp.StandardizedTransformer.steer``'s API: ``None``
    means "every position", an ``int`` or ``list[int]`` is uniform across the
    batch (additive mode only). Per-row position resolution
    (``TokenSelector``-driven) is future work.
    """

    layers: list[int]
    vector: torch.Tensor
    factor: float = 1.0
    mode: SteeringMode = "additive"
    positions: int | list[int] | None = None


__all__ = ["PendingSteer", "SteeringMode"]
