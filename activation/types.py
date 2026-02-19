"""Shared data types for the activation package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import TypedDict


class ModelMetadata(TypedDict):
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int


@dataclass(frozen=True)
class LayerSpec:
    kind: str
    value: int | str | None = None


class ExtractionResult(TypedDict):
    model: ModelMetadata
    requested: list[str]
    activations: dict[str, torch.Tensor]
    sample_ids: list[str]
    labels: list[int | None]
    storage: dict[str, Any]
