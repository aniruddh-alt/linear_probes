"""Sweep configuration parameters for layerwise probe sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig


@dataclass
class SweepParams(BaseConfig):
    """Configuration for layerwise probe sweep orchestration."""

    activation_targets: list | None = None
    batch_size: int = 32
    selection_metric: str = "auroc"
    maximize_metric: bool = True
    control_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    enforce_control_sanity: bool = True

    def __post_init__(self) -> None:
        if self.activation_targets is not None:
            if not self.activation_targets:
                raise ValueError("activation_targets must be non-empty when provided.")
            for target in self.activation_targets:
                if isinstance(target, int):
                    continue
                if isinstance(target, str) and target.strip():
                    continue
                raise ValueError(
                    "activation_targets items must be int layer indices "
                    "or non-empty activation key strings."
                )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if not self.selection_metric.strip():
            raise ValueError("selection_metric must be non-empty.")
        if not self.control_seeds:
            raise ValueError("control_seeds must be non-empty.")
        if any(not isinstance(seed, int) for seed in self.control_seeds):
            raise ValueError("control_seeds must contain only integers.")
