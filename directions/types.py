"""Shared data types for the directions package."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffMeansLayerResult:
    key: str
    direction: torch.Tensor  # normalized
    raw_norm: float  # ||r_l|| before normalization
    positive_count: int
    negative_count: int
    val_metrics: dict[str, float | tuple[float, float]]


@dataclass
class DiffMeansSweepResult:
    layers: dict[str, DiffMeansLayerResult]
    best_key: str
    best_metric: str
    best_score: float
    best_direction: torch.Tensor
    test_metrics: dict[str, float | tuple[float, float]]
    controls: dict[str, dict[str, float]]
    split_sizes: tuple[int, int, int]
    dataset_fingerprint: str
    manifest_path: str | None
