"""PyTorch dataset helpers for linear probing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from sonde.activation.storage import (
    load_activation_value,
    load_extraction_manifest,
    resolve_activation_key,
)


class ProbingDataset(
    Dataset[
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
):
    """Dataset for binary probes supporting pooled (N, D) and sequence (variable S, D) modes."""

    def __init__(
        self,
        features: Sequence[torch.Tensor] | torch.Tensor,
        labels: Sequence[int],
        sequence_mode: bool | None = None,
    ):
        label_values = [int(label) for label in labels]
        if any(label not in (0, 1) for label in label_values):
            raise ValueError("Binary probe labels must be 0 or 1.")
        self.labels = torch.tensor(label_values, dtype=torch.long)

        self.sequence_mode = False
        self._sequence_features: list[torch.Tensor] | None = None

        if (
            isinstance(features, list)
            and features
            and isinstance(features[0], torch.Tensor)
            and features[0].ndim == 2
        ):
            hidden_dim = features[0].shape[1]
            all_2d_same_hidden = all(
                f.ndim == 2 and f.shape[1] == hidden_dim for f in features
            )
            # A list of 2-D (S, D) tensors is per-token sequence data. Default to
            # sequence mode so equal-length sequences are NOT silently flattened
            # to (N, S*D) — a correctness landmine that trains a probe on the
            # wrong input dimensionality. Pass sequence_mode=False to explicitly
            # opt into pooled flattening instead.
            if all_2d_same_hidden and sequence_mode is not False:
                self.sequence_mode = True
                self._sequence_features = [f.detach().float() for f in features]
                if len(self._sequence_features) != len(labels):
                    raise ValueError(
                        "features and labels must have same length, got "
                        f"{len(self._sequence_features)} and {len(labels)}."
                    )
                self.features = torch.empty(0)
                return

        features_tensor = self._as_feature_matrix(features)
        if int(features_tensor.shape[0]) != len(labels):
            raise ValueError(
                "features and labels must have same length, got "
                f"{int(features_tensor.shape[0])} and {len(labels)}."
            )
        self.features = features_tensor

    def __len__(self) -> int:
        if self.sequence_mode and self._sequence_features is not None:
            return len(self._sequence_features)
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self.sequence_mode and self._sequence_features is not None:
            feat = self._sequence_features[index]
            mask = torch.ones(feat.shape[0], dtype=torch.float32)
            return feat, self.labels[index], mask
        return self.features[index], self.labels[index]

    @classmethod
    def from_extraction_result(
        cls,
        extraction: Mapping[str, Any],
        *,
        activation_key: str,
        labels: Sequence[int] | None = None,
        positive_indices: Iterable[int] | None = None,
    ) -> ProbingDataset:
        """Build dataset from one activation stream in an ExtractionResult."""
        storage = extraction.get("storage")
        if activation_key not in extraction["activations"] and not (
            isinstance(storage, dict) and storage.get("mode") == "safetensors"
        ):
            keys = ", ".join(extraction["activations"].keys())
            raise KeyError(
                f"activation_key '{activation_key}' not found. Available keys: {keys}"
            )

        raw_features = cls._resolve_raw_features(
            extraction, activation_key=activation_key
        )

        if isinstance(raw_features, list):
            num_features = len(raw_features)
        else:
            num_features = int(raw_features.shape[0])

        sample_ids = extraction.get("sample_ids")
        if sample_ids is not None and len(sample_ids) != num_features:
            raise ValueError(
                "extraction sample_ids length does not match features length: "
                f"{len(sample_ids)} vs {num_features}."
            )

        if labels is None:
            extraction_labels = extraction.get("labels")
            if extraction_labels is not None:
                if len(extraction_labels) != num_features:
                    raise ValueError(
                        "extraction labels length does not match features length: "
                        f"{len(extraction_labels)} vs {num_features}."
                    )
                if any(label is None for label in extraction_labels):
                    raise ValueError(
                        "extraction contains unlabeled samples. Pass explicit `labels` "
                        "or provide labels when building samples."
                    )
                labels = [int(label) for label in extraction_labels]

        if labels is None:
            if positive_indices is None:
                raise ValueError(
                    "Cannot build a binary ProbingDataset without labels. Pass "
                    "`labels=`, pass `positive_indices=`, or include a `labels` "
                    "field in the extraction. (Previously this silently labelled "
                    "every sample 0.)"
                )
            labels = [0] * num_features
            for idx in positive_indices:
                if idx < 0 or idx >= num_features:
                    raise IndexError(
                        f"positive index {idx} out of range for {num_features} samples."
                    )
                labels[idx] = 1

        if isinstance(raw_features, list):
            return cls(features=raw_features, labels=labels)
        return cls(features=cls._as_feature_matrix(raw_features), labels=labels)

    @classmethod
    def from_extraction_path(
        cls,
        extraction_path: str | Path,
        *,
        activation_key: str | None = None,
        labels: Sequence[int] | None = None,
        positive_indices: Iterable[int] | None = None,
        map_location: str | torch.device = "cpu",
    ) -> ProbingDataset:
        extraction = load_extraction_manifest(
            Path(extraction_path), map_location=map_location
        )
        resolved_key = resolve_activation_key(extraction, activation_key)
        return cls.from_extraction_result(
            extraction,
            activation_key=resolved_key,
            labels=labels,
            positive_indices=positive_indices,
        )

    @staticmethod
    def _flatten_feature(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 0:
            return feature.reshape(1)
        return feature.reshape(-1)

    @classmethod
    def _as_feature_matrix(
        cls, features: Sequence[torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            tensor = features.detach().float()
            if tensor.ndim == 0:
                return tensor.reshape(1, 1)
            if tensor.ndim == 1:
                return tensor.reshape(-1, 1)
            return tensor.reshape(tensor.shape[0], -1)

        flat_features = [
            cls._flatten_feature(feature).detach().float() for feature in features
        ]
        if not flat_features:
            return torch.empty((0, 0), dtype=torch.float32)
        expected_dim = flat_features[0].numel()
        for idx, feature in enumerate(flat_features):
            if feature.numel() != expected_dim:
                raise ValueError(
                    "All features must flatten to the same dimensionality, got "
                    f"{feature.numel()} at index {idx} vs expected {expected_dim}."
                )
        return torch.stack(flat_features, dim=0)

    @classmethod
    def _resolve_raw_features(
        cls, extraction: Mapping[str, Any], *, activation_key: str
    ) -> torch.Tensor | list[torch.Tensor]:
        """Resolve features without forcing to matrix -- preserves list-of-2D for sequence mode."""
        raw_features = load_activation_value(
            extraction,
            activation_key=activation_key,
            map_location="cpu",
        )
        if isinstance(raw_features, torch.Tensor):
            return raw_features
        if isinstance(raw_features, (list, Sequence)):
            return list(raw_features)
        raise TypeError(
            f"Unsupported activation value format for key '{activation_key}'."
        )

    @classmethod
    def _resolve_activation_tensor(
        cls, extraction: Mapping[str, Any], *, activation_key: str
    ) -> torch.Tensor:
        """Legacy method - always returns flattened tensor."""
        raw_features = load_activation_value(
            extraction,
            activation_key=activation_key,
            map_location="cpu",
        )
        if isinstance(raw_features, torch.Tensor):
            return cls._as_feature_matrix(raw_features)
        if isinstance(raw_features, Sequence):
            return cls._as_feature_matrix(raw_features)
        raise TypeError(
            f"Unsupported activation value format for key '{activation_key}'."
        )
