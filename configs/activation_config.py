"""Configuration objects for activation extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .model_config import ModelConfig


@dataclass(frozen=True)
class ActivationConfig:
    """Typed configuration for activation extraction runs.

    Attributes:
        model_config: Configuration for model initialization.
        save_path: Path to save the extracted activations.
        activations: List of activation names to extract.
        batch_size: Batch size for activation extraction.
        token_index: Index of the token to extract activations for.
        remote: Whether to extract activations from a remote model.
        to_cpu: Whether to move activations to CPU.
    """

    model_config: ModelConfig
    save_path: str | Path
    activations: list[str] = field(default_factory=list)
    batch_size: int = 8
    token_index: int | None = -1
    remote: bool = False
    to_cpu: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
