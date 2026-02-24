"""Probe training configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.configs.base import BaseConfig


@dataclass
class ProbeParams(BaseConfig):
    """Training configuration for linear probes.

    Maps from the existing ProbeConfig in configs/types.py.
    """

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_grad_norm: Optional[float] = None
    threshold: float = 0.5
    device: Optional[str] = None
    seed: Optional[int] = None
    early_stopping_patience: Optional[int] = 5
    early_stopping_min_delta: float = 1e-4
    bootstrap_samples: int = 0
    bootstrap_confidence: float = 0.95
