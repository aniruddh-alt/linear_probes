"""Probe training configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.configs.base import BaseConfig


@dataclass
class ProbeParams(BaseConfig):
    """Training configuration for linear probes.

    Typed configuration for linear probe training.
    """

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    l1_weight: float = 0.0
    pca_components: Optional[int] = None
    max_grad_norm: Optional[float] = None
    threshold: float = 0.5
    device: Optional[str] = None
    seed: Optional[int] = None
    early_stopping_patience: Optional[int] = 5
    early_stopping_min_delta: float = 1e-4
    bootstrap_samples: int = 0
    bootstrap_confidence: float = 0.95
    probe_type: str = "linear"
    probe_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.l1_weight < 0:
            raise ValueError("l1_weight must be >= 0.")
        if self.pca_components is not None and self.pca_components <= 0:
            raise ValueError("pca_components must be > 0 when provided.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when provided.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1).")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 0
        ):
            raise ValueError("early_stopping_patience must be >= 0 when provided.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1).")
