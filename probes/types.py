"""Shared data types for the probes package."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from probes.linear import BinaryLinearProbeTrainer


@dataclass
class TrainedLayerProbe:
    activation_key: str
    trainer: BinaryLinearProbeTrainer
    history: dict[str, list[float | tuple[float, float]]]
    metrics: dict[str, float | tuple[float, float]]
    direction: torch.Tensor
    bias: float


@dataclass
class LayerProbeSweepResult:
    probes: dict[str, TrainedLayerProbe]
    best_key: str
    best_metric: str
    best_score: float
    best_direction: torch.Tensor
    best_bias: float
    split_sizes: tuple[int, int]
