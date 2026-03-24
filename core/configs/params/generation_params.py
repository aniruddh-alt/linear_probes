"""Generation configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass

from core.configs.base import BaseConfig


@dataclass
class GenerationParams(BaseConfig):
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
