"""InterventionContext — a reusable buffer of steering operations for nnterp.

It buffers ``add_steering`` calls and replays them via :meth:`apply`, which must
be invoked **inside a literal** ``with model.trace(...)`` or
``with model.generate(...)`` block. nnsight discovers traced operations by
introspecting that block's source, so steers cannot be applied from a wrapper
context manager — they must be issued from within the user's own ``with`` block.
This constraint is verified against gpt2 (see ``docs/intervention_design.md``).

Supported modes: ``additive`` (add ``factor·v̂``) and ``project_subtract``
(directional ablation ``h - factor·(h·v̂)v̂``). Both reapply on every decoded
token when used inside ``model.generate``.

Usage::

    from sonde.interventions import InterventionContext

    # Read logits under an additive steer.
    ctx = InterventionContext(model).add_steering(layers=[1, 3], vector=v, factor=0.5)
    with model.trace("The weather today is"):
        ctx.apply()
        logits = model.logits.save()

    # Ablate a probe's direction during generation (the causal probe test).
    ctx = InterventionContext(model).add_steering(
        layers=[probe.layer], vector=probe.direction, mode="project_subtract"
    )
    with model.generate(prompts, max_new_tokens=128) as tracer:
        ctx.apply()
        out = model.generator.output.save()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .steering import apply_pending_steers
from .types import PendingSteer, SteeringMode
from .vectors import load_vector

if TYPE_CHECKING:
    from sonde.core.configs.params.steering_params import SteeringParams


class InterventionContext:
    """A reusable buffer of steering operations over an nnterp model."""

    def __init__(self, model: Any):
        self.model = model
        self._steers: list[PendingSteer] = []

    # ── configuration ─────────────────────────────────────────────────────

    def add_steering(
        self,
        layers: int | list[int] | str,
        vector: Any,
        *,
        mode: SteeringMode = "additive",
        factor: float = 1.0,
        normalize: bool = True,
        positions: int | list[int] | None = None,
        vector_key: str = "",
    ) -> InterventionContext:
        """Buffer a steering operation. Returns ``self`` for chaining.

        Args:
            layers: Layer index, list of indices, or ``"all"`` (expands to
                ``range(model.num_layers)`` at configure time).
            vector: Source resolvable by :func:`sonde.interventions.load_vector`
                — a tensor, a path, a probe, or a direction result.
            mode: ``"additive"`` or ``"project_subtract"`` (directional ablation).
            factor: meaning depends on ``mode``. ``additive``: signed steering
                strength (activation units; ``0`` is a no-op). ``project_subtract``:
                ablation fraction (``1.0`` fully removes the direction).
            normalize: L2-normalise the vector before storage.
            positions: ``None`` (every position), ``int``, or ``list[int]``
                (uniform across batch). Supported for ``additive`` only;
                ``project_subtract`` ablates every position.
            vector_key: key for multi-tensor ``.safetensors`` / ``.pt`` sources.
        """
        if mode not in ("additive", "project_subtract"):
            raise ValueError(
                f"Unknown steering mode '{mode}'. Expected 'additive' or "
                "'project_subtract'."
            )
        resolved_layers = self._resolve_layers(layers)
        resolved_vector = load_vector(vector, key=vector_key, normalize=normalize)
        self._steers.append(
            PendingSteer(
                layers=resolved_layers,
                vector=resolved_vector,
                factor=float(factor),
                mode=mode,
                positions=positions,
            )
        )
        return self

    def clear(self) -> InterventionContext:
        """Forget every pending intervention. Returns ``self``."""
        self._steers.clear()
        return self

    @property
    def pending_steers(self) -> list[PendingSteer]:
        """Inspect the buffered steering operations (read-only by convention)."""
        return list(self._steers)

    # ── execution ─────────────────────────────────────────────────────────

    def apply(self) -> None:
        """Replay every pending intervention against the model.

        Call this **inside** a literal ``with model.trace(...)`` or
        ``with model.generate(...)`` block.
        """
        apply_pending_steers(self.model, self._steers)

    # ── classmethod constructors ──────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        model: Any,
        steering: SteeringParams | list[SteeringParams] | None = None,
    ) -> InterventionContext:
        """Build a context from declarative ``SteeringParams`` blocks.

        ``steering`` may be a single block or a list (added in order). Blocks
        with ``enabled=False`` are skipped.
        """
        ctx = cls(model)
        if steering is None:
            return ctx
        blocks = steering if isinstance(steering, list) else [steering]
        for block in blocks:
            if not getattr(block, "enabled", True):
                continue
            ctx.add_steering(
                layers=block.layers,
                vector=block.vector_path,
                vector_key=block.vector_key,
                # block.mode is validated against the same set by SteeringParams.
                mode=cast(SteeringMode, block.mode),
                # SteeringParams.factor resolves strength + the mode-aware default.
                factor=block.factor,
                normalize=block.normalize,
            )
        return ctx

    # ── internals ─────────────────────────────────────────────────────────

    def _resolve_layers(self, layers: int | list[int] | str) -> list[int]:
        if isinstance(layers, str):
            if layers.lower() != "all":
                raise ValueError(f"layers string must be 'all'; got {layers!r}.")
            num_layers = int(getattr(self.model, "num_layers", 0))
            if num_layers <= 0:
                raise ValueError(
                    "InterventionContext.add_steering(layers='all') requires "
                    "`model.num_layers` to be a positive int."
                )
            return list(range(num_layers))
        if isinstance(layers, int):
            return [int(layers)]
        if isinstance(layers, (list, tuple)):
            resolved = [int(layer) for layer in layers]
            if not resolved:
                raise ValueError("layers list cannot be empty.")
            return resolved
        raise TypeError(
            f"layers must be int, list[int], or 'all'; got {type(layers).__name__}."
        )


__all__ = ["InterventionContext"]
