"""Steering configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

from sonde.core.configs.base import BaseConfig

_VALID_MODES = frozenset({"project_subtract", "additive"})


@dataclass
class SteeringParams(BaseConfig):
    """Configuration for activation steering during generation.

    ``strength`` is the canonical YAML/config field; downstream callers that
    speak nnterp's vocabulary read it via the ``factor`` property on this
    dataclass so the steering API stays consistent across config and code.

    The default ``strength`` is **mode-dependent** because the two modes use the
    knob differently: ``additive`` uses it as a steering strength in activation
    units (default ``10.0``), while ``project_subtract`` uses it as an ablation
    fraction (default ``1.0`` = full directional ablation). Leave ``strength``
    unset to get the right default for the chosen mode.
    """

    enabled: bool = False
    vector_path: str = ""
    vector_key: str = ""
    layers: list[int] = field(default_factory=list)
    strength: float | None = None
    mode: str = "project_subtract"
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid steering mode '{self.mode}'. "
                f"Expected one of: {', '.join(sorted(_VALID_MODES))}"
            )
        if self.strength is None:
            # Mode-aware default: a 10.0 additive default would 10x-over-ablate
            # (and sign-flip) in project_subtract mode.
            self.strength = 1.0 if self.mode == "project_subtract" else 10.0
        if self.enabled:
            if not self.vector_path:
                raise ValueError("vector_path is required when steering is enabled.")
            if not self.layers:
                raise ValueError("layers is required when steering is enabled.")

    @property
    def factor(self) -> float:
        """Alias for ``strength`` matching nnterp's ``steer(factor=...)`` API."""
        return self.strength if self.strength is not None else 1.0
