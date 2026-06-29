"""Data splitting configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass

from sonde.core.configs.base import BaseConfig


@dataclass
class SplitParams(BaseConfig):
    """Configuration for train/val/test splitting."""

    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int | None = 0
    auto_group_by_id_when_none: bool = True

    def __post_init__(self) -> None:
        for name in ("train_fraction", "val_fraction", "test_fraction"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0 (got {total:.6f})")
