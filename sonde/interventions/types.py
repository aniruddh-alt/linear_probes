"""Dataclasses representing pending interventions on a transformer model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PendingSteer:
    """A configured-but-not-yet-applied steering operation.

    Vectors are stored normalised (unit L2) when ``normalize=True`` was used at
    construction time; ``factor`` carries the signed strength so additive and
    project-subtract modes share a single vector representation.

    ``positions`` mirrors ``nnterp.StandardizedTransformer.steer``'s API: ``None``
    means "every position", an ``int`` or ``list[int]`` is uniform across the
    batch. Per-row position resolution (``TokenSelector``-driven) is deferred to
    R-2.
    """

    layers: list[int]
    vector: torch.Tensor
    factor: float = 1.0
    mode: str = "additive"
    positions: int | list[int] | None = None


__all__ = ["PendingSteer"]
