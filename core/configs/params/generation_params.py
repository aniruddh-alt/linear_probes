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
