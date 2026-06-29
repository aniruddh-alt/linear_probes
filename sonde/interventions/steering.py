"""Apply pending additive steering operations to a model inside a trace.

R-1 scope is additive steering only — it forwards to nnterp's native
``StandardizedTransformer.steer``, which writes ``factor · vector`` into
``layers_output[L]`` (the post-block residual) for every position or the listed
positions. Directional ablation (``project_subtract``) and ``TokenSelector``-
driven per-row positions land in R-2.
"""

from __future__ import annotations

from typing import Any

from .types import PendingSteer


def apply_pending_steers(model: Any, pending: list[PendingSteer]) -> None:
    """Replay each PendingSteer through ``model.steer(...)`` inside an active
    trace context. Call sites are responsible for being inside ``model.trace``
    or ``model.generate`` when this runs.
    """
    for steer in pending:
        if steer.mode != "additive":
            raise NotImplementedError(
                f"Steering mode '{steer.mode}' is not yet implemented "
                "(R-1 supports 'additive' only; 'project_subtract' lands in R-2)."
            )
        model.steer(
            layers=list(steer.layers),
            steering_vector=steer.vector,
            factor=steer.factor,
            positions=steer.positions,
        )


__all__ = ["apply_pending_steers"]
