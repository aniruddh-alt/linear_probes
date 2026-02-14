"""PyTorch dataset helpers for linear probing."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class ProbingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple (activation, label) dataset for binary linear probes."""

    def __init__(self, features: Sequence[torch.Tensor], labels: Sequence[int]):
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have same length, got {len(features)} and {len(labels)}."
            )
        self.features = [feature.float() for feature in features]
        self.labels = [int(label) for label in labels]
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("Binary probe labels must be 0 or 1.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], torch.tensor(self.labels[index], dtype=torch.long)

    @classmethod
    def from_extraction_result(
        cls,
        extraction: dict,
        *,
        activation_key: str,
        labels: Sequence[int] | None = None,
        positive_indices: Iterable[int] | None = None,
    ) -> "ProbingDataset":
        """Build dataset from one activation stream in an ExtractionResult.

        If labels are not provided, all samples default to 0.
        Optionally pass positive_indices to flip selected samples to 1.
        """
        if activation_key not in extraction["activations"]:
            keys = ", ".join(extraction["activations"].keys())
            raise KeyError(
                f"activation_key '{activation_key}' not found. Available keys: {keys}"
            )

        raw_features = extraction["activations"][activation_key]
        features = [cls._flatten_feature(tensor) for tensor in raw_features]

        if labels is None:
            labels = [0] * len(features)
            if positive_indices is not None:
                labels = list(labels)
                for idx in positive_indices:
                    if idx < 0 or idx >= len(features):
                        raise IndexError(
                            f"positive index {idx} out of range for {len(features)} samples."
                        )
                    labels[idx] = 1
        return cls(features=features, labels=labels)

    @staticmethod
    def _flatten_feature(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 0:
            return feature.reshape(1)
        return feature.reshape(-1)
