"""Steering configuration parameters."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig

_VALID_MODES = frozenset({"project_subtract", "additive"})


@dataclass
class SteeringParams(BaseConfig):
    """Configuration for activation steering during generation."""

    enabled: bool = False
    vector_path: str = ""
    vector_key: str = ""
    layers: list[int] = field(default_factory=list)
    strength: float = 10.0
    mode: str = "project_subtract"
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid steering mode '{self.mode}'. "
                f"Expected one of: {', '.join(sorted(_VALID_MODES))}"
            )
        if self.enabled:
            if not self.vector_path:
                raise ValueError("vector_path is required when steering is enabled.")
            if not self.layers:
                raise ValueError("layers is required when steering is enabled.")
