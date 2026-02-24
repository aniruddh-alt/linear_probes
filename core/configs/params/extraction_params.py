"""Activation extraction configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.configs.base import BaseConfig


@dataclass
class ExtractionParams(BaseConfig):
    """Configuration for activation extraction runs.

    Maps from the existing ActivationConfig in configs/types.py.
    """

    save_path: str = ""
    activations: list[str] = field(default_factory=list)
    batch_size: int = 8
    token_index: Optional[int] = -1
    remote: bool = False
    to_cpu: bool = True
