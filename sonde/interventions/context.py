"""InterventionContext — unified read+write context over an nnterp model.

R-1 scope: additive steering only. The context buffers ``add_steering`` calls
and replays them inside ``model.trace`` / ``model.generate``. Per-position
``TokenSelector`` integration, ``project_subtract`` mode, activation patching,
and per-head steering land in subsequent PRs (R-2 onward).

Usage::

    from interventions import InterventionContext

    # Programmatic: ad-hoc
    ctx = InterventionContext(model)
    with ctx.trace("The weather today is"):
        ctx.add_steering(layers=[1, 3], vector=v, factor=0.5)
        logits = model.logits.save()

    # Programmatic: configured once, applied across many traces
    ctx = (
        InterventionContext(model)
        .add_steering(layers=[15], vector=probe.direction, factor=1.0)
    )
    with ctx:
        out = ctx.generate(prompts, max_new_tokens=128)
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from .steering import apply_pending_steers
from .types import PendingSteer
from .vectors import load_vector

if TYPE_CHECKING:
    from sonde.core.configs.params.steering_params import SteeringParams


class InterventionContext(AbstractContextManager):
    """Unified read+write context over an nnterp StandardizedTransformer.

    Interventions are configured up-front via the chainable ``add_*`` methods
    and applied inside ``trace(...)`` or ``generate(...)``. The same context
    can be reused across multiple traces, or rebuilt with ``clear()``.
    """

    def __init__(self, model: Any):
        self.model = model
        self._steers: list[PendingSteer] = []

    # ── configuration ─────────────────────────────────────────────────────

    def add_steering(
        self,
        layers: int | list[int] | str,
        vector: Any,
        *,
        mode: str = "additive",
        factor: float = 1.0,
        normalize: bool = True,
        positions: int | list[int] | None = None,
        vector_key: str = "",
    ) -> InterventionContext:
        """Buffer a steering operation. Returns ``self`` for chaining.

        Args:
            layers: Layer index, list of indices, or the string ``"all"`` to
                expand to ``range(model.num_layers)`` at apply time.
            vector: Source resolvable by :func:`interventions.vectors.load_vector`
                — a tensor, a path string, a probe instance, or a direction
                result. Loaded once at configure time.
            mode: ``"additive"`` (R-1). ``"project_subtract"`` raises until R-2.
            factor: Signed strength multiplier. ``factor=0`` is a no-op.
            normalize: If True, the vector is L2-normalised before storage.
            positions: Position selector compatible with ``nnterp.steer`` —
                ``None`` for every position, ``int`` or ``list[int]`` for a
                uniform-across-batch subset. ``TokenSelector`` support lands
                in R-2.
            vector_key: Forwarded to ``load_vector`` for ``.safetensors`` /
                ``.pt`` files containing multiple tensors.
        """
        if mode not in ("additive", "project_subtract"):
            raise ValueError(
                f"Unknown steering mode '{mode}'. Expected 'additive' or "
                "'project_subtract'."
            )
        resolved_layers = self._resolve_layers(layers)
        resolved_vector = load_vector(
            vector,
            key=vector_key,
            normalize=normalize,
        )
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

    def trace(self, *args: Any, **kwargs: Any) -> _AppliedTrace:
        """Return a context manager equivalent to ``model.trace(...)`` that
        applies all pending interventions inside the trace.
        """
        inner = self.model.trace(*args, **kwargs)
        return _AppliedTrace(inner, on_enter=self._apply)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Run ``model.generate(...)`` under the configured interventions.

        Interventions are applied inside the underlying generation tracer so
        they fire on every decoded token automatically.
        """
        tracer = self.model.generate(*args, **kwargs)
        with tracer:
            self._apply()
        return tracer

    def __enter__(self) -> InterventionContext:  # for `with ctx:` ergonomics
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    # ── classmethod constructors ──────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        model: Any,
        steering: SteeringParams | list[SteeringParams] | None = None,
    ) -> InterventionContext:
        """Build a context from declarative ``SteeringParams`` blocks.

        ``steering`` may be a single :class:`SteeringParams` or a list (each is
        added as a separate operation; order is preserved). Blocks with
        ``enabled=False`` are skipped.
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
                mode=block.mode,
                factor=_block_factor(block),
                normalize=block.normalize,
            )
        return ctx

    # ── internals ─────────────────────────────────────────────────────────

    def _apply(self) -> None:
        """Replay every pending intervention. Caller must be inside a trace."""
        apply_pending_steers(self.model, self._steers)

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
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"layers must be int, list[int], or 'all'; got {type(layers).__name__}."
        )


class _AppliedTrace(AbstractContextManager):
    """Wraps an nnterp/nnsight trace context manager so that pending
    interventions are applied immediately after entering the trace.
    """

    def __init__(self, inner: AbstractContextManager, *, on_enter: Any):
        self._inner = inner
        self._on_enter = on_enter

    def __enter__(self) -> Any:
        result = self._inner.__enter__()
        self._on_enter()
        return result

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)


def _block_factor(block: Any) -> float:
    """Resolve the steering strength on a SteeringParams block.

    Reads ``block.factor`` (a property aliasing ``strength``) if present, falling
    back to ``block.strength`` so any duck-typed config object with either name
    works.
    """
    value = getattr(block, "factor", None)
    if value is not None:
        return float(value)
    return float(getattr(block, "strength", 1.0))


__all__ = ["InterventionContext"]
