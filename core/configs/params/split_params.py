"""Data splitting configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.configs.base import BaseConfig


@dataclass
class SplitParams(BaseConfig):
    """Configuration for train/val/test splitting.

    Maps from the split-related fields in LayerProbeSweepConfig.
    """

    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: Optional[int] = 0
    auto_group_by_id_when_none: bool = True
