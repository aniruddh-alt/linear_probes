"""Activation extraction configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig


@dataclass
class ExtractionParams(BaseConfig):
    """Configuration for activation extraction runs.

    Typed configuration for activation extraction runs.
    """

    save_path: str = ""
    activations: list[str] = field(default_factory=list)
    batch_size: int = 8
    token_index: int | None = -1
    remote: bool = False
    to_cpu: bool = True
