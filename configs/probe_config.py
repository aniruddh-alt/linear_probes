"""Configuration for training linear probes.

This module defines the training hyperparameters and settings used
for fitting linear probes on model activations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTrainConfig:
    """Training configuration for linear probes.

    Attributes:
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        weight_decay: L2 regularization coefficient.
        threshold: Classification threshold for binary predictions.
        device: Device to train on (e.g., 'cuda', 'cpu'). If None, auto-detected.
        seed: Optional random seed for deterministic initialization.
        bootstrap_samples: Number of bootstrap resamples for evaluation CIs.
        bootstrap_confidence: Confidence level for bootstrap intervals.
    """

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    threshold: float = 0.5
    device: str | None = None
    seed: int | None = None
    bootstrap_samples: int = 0
    bootstrap_confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1).")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1).")
