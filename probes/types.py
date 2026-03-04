"""Shared data types for the probes package."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from probes.linear import BinaryProbeTrainer


@dataclass
class TrainedLayerProbe:
    activation_key: str
    trainer: BinaryProbeTrainer
    history: dict[str, list[float | tuple[float, float]]]
    val_metrics: dict[str, float | tuple[float, float]]
    direction: torch.Tensor | None = None
    bias: float | None = None


@dataclass
class LayerProbeSweepResult:
    probes: dict[str, TrainedLayerProbe]
    best_key: str
    best_metric: str
    best_score: float
    test_metrics: dict[str, float | tuple[float, float]]
    controls: dict[str, dict[str, float]]
    best_direction: torch.Tensor | None = None
    best_bias: float | None = None
    split_sizes: tuple[int, int, int] = (0, 0, 0)
    dataset_fingerprint: str = ""
    manifest_path: str | None = None
