"""Base class for all probes.

Encodes the Kramár & Engels et al. (2026) 6-stage framework:

    (1) input  (B, S, D)  or  (B, D)
    (2) per-position TRANSFORM     -> (B, S, D')
    (3) per-position SCORE         -> (B, S, num_classes)  or pass-through (B, S, D')
    (4)+(5) AGGREGATE over positions -> (B, num_classes)
    (6) scalar output for each class

Subclasses override ``transform`` (default identity), ``score`` and ``aggregate``.
The base class handles the (B, S, D) <-> (B, D) dispatch and mask normalisation
once, and exposes ``direction`` / ``bias`` properties so downstream code (the
sweep runner, the steering pipeline, weight-orthogonalisation routines) can
treat any probe uniformly without reaching into ``model.linear.weight``.

Reference: ``docs/toolkit_audit.md`` §6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseProbe(nn.Module, ABC):
    """Common base class for every probe architecture in this toolkit."""

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        self.input_dim = input_dim
        self.num_classes = num_classes

    # ────────────────────────────── Forward dispatch ──────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, D) or (B, S, D) -> (B, num_classes)."""
        x_3d, mask_3d = self._prepare_input(x, mask)
        h = self.transform(x_3d)
        scores = self.score(h)
        return self.aggregate(scores, h, mask_3d)

    @staticmethod
    def _prepare_input(
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalise input to (B, S, D) and synthesise a trivial mask for 2D input."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
            mask = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        elif x.ndim != 3:
            raise ValueError(
                f"Probe expected (B, D) or (B, S, D), got tensor with shape {tuple(x.shape)}."
            )
        return x, mask

    # ────────────────────────────── Stage hooks ──────────────────────────────

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 2: per-position transform. Default identity.

        Override to insert an MLP / projection / per-position nonlinearity before scoring.
        Input shape: (B, S, D). Output shape: (B, S, D').
        """
        return x

    @abstractmethod
    def score(self, h: torch.Tensor) -> torch.Tensor:
        """Stage 3: per-position scoring.

        Typically returns (B, S, num_classes). Probes that aggregate raw activations
        first (e.g. ``MaxProbe``) may return ``h`` unchanged and let ``aggregate``
        apply the readout.
        """

    @abstractmethod
    def aggregate(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Stages 4-5: pool over the sequence dimension to (B, num_classes)."""

    # ────────────────────────────── Causal-interp hooks ──────────────────────────────

    @property
    def direction(self) -> torch.Tensor | None:
        """Unit-norm direction in activation space, if the probe defines one.

        Default: if the subclass has a ``self.linear: nn.Linear`` readout with
        ``num_classes == 1`` and weight shape ``(1, D)``, return its row vector
        normalised to unit L2. Otherwise return ``None`` (e.g. for multi-head
        attention probes, which do not have a single causal direction).

        Used by steering / weight-orthogonalisation pipelines.
        """
        linear = getattr(self, "linear", None)
        if not isinstance(linear, nn.Linear):
            return None
        weight = linear.weight.detach().cpu().reshape(-1).float()
        if weight.numel() != self.input_dim:
            return None
        norm = float(torch.linalg.vector_norm(weight).item())
        if norm == 0.0:
            return weight
        return weight / norm

    @property
    def bias(self) -> float | None:
        """Scalar bias of the decision function, if defined.

        Default: pulls ``self.linear.bias`` when it exists and is a scalar.
        Returns ``None`` for biases that are vectors (e.g. multi-class readouts).
        """
        linear = getattr(self, "linear", None)
        if not isinstance(linear, nn.Linear) or linear.bias is None:
            return None
        bias = linear.bias.detach()
        if bias.numel() != 1:
            return None
        return float(bias.cpu().item())


__all__ = ["BaseProbe"]
