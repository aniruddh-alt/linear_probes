"""Apply pending steering operations to a model inside a trace.

Two modes are supported:

- ``additive`` forwards to nnterp's native ``StandardizedTransformer.steer``,
  which adds ``factor * vector`` to ``layers_output[L]`` (the post-block
  residual) at every position (or the listed positions).
- ``project_subtract`` performs directional ablation
  ``h' = h - factor * (h . v̂) v̂`` by writing directly into the
  ``layers_output[L]`` proxy. With ``factor=1`` it removes the component of the
  residual along ``v̂`` at every position — the canonical "ablate this concept
  direction" causal test (Arditi et al. 2024).

The ablation math lives in :func:`directional_ablation`, a pure tensor function
that is unit-tested independently of any model. Both write paths run *inside* an
active ``model.trace`` / ``model.generate`` context — call sites are responsible
for that.
"""

from __future__ import annotations

from typing import Any

import torch

from .types import PendingSteer


def directional_ablation(
    h: torch.Tensor, v: torch.Tensor, factor: float = 1.0
) -> torch.Tensor:
    """Return ``h - factor * (h . v̂) v̂`` along the last dim.

    ``v`` is cast to ``h``'s dtype/device. With ``factor=1`` and a unit ``v`` the
    component of ``h`` along ``v`` is removed (its projection becomes ~0).
    """
    v = v.to(dtype=h.dtype, device=h.device)
    dot = (h * v).sum(dim=-1, keepdim=True)
    return h - factor * dot * v


def apply_pending_steers(model: Any, pending: list[PendingSteer]) -> None:
    """Replay each :class:`PendingSteer` against ``model`` inside an active trace.

    Args:
        model: an nnterp ``StandardizedTransformer`` (or compatible) currently
            inside a ``trace``/``generate`` context.
        pending: the buffered steering operations, applied in order.
    """
    for steer in pending:
        if steer.mode == "additive":
            model.steer(
                layers=list(steer.layers),
                steering_vector=steer.vector,
                factor=steer.factor,
                positions=steer.positions,
            )
        elif steer.mode == "project_subtract":
            if steer.positions is not None:
                raise NotImplementedError(
                    "project_subtract currently ablates every position; explicit "
                    "`positions` (per-row TokenSelector masks) are not yet "
                    "supported for directional ablation. Use positions=None."
                )
            for layer in steer.layers:
                h = model.layers_output[layer]
                model.layers_output[layer] = directional_ablation(
                    h, steer.vector, steer.factor
                )
        else:
            raise NotImplementedError(
                f"Steering mode '{steer.mode}' is not supported "
                "(expected 'additive' or 'project_subtract')."
            )


__all__ = ["apply_pending_steers", "directional_ablation"]
