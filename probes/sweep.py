"""High-level orchestrator for layerwise linear probe sweeps."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from activation.types import ExtractionResult
from configs import LayerProbeSweepConfig
from dataset import ProbingDataset
from probes.linear import BinaryLinearProbeTrainer
from probes.types import TrainedLayerProbe


class LayerProbeSweepRunner:
    """Runs train/validation split + fit/eval for multiple activation layers."""

    def __init__(self, config: LayerProbeSweepConfig | None = None):
        self.config = config or LayerProbeSweepConfig()

    def run(
        self,
        extraction: ExtractionResult | dict[str, Any],
        *,
        labels: list[int] | None = None,
        positive_indices: list[int] | None = None,
    ) -> dict[str, TrainedLayerProbe]:
        activation_keys = self._resolve_activation_keys(extraction)
        if not activation_keys:
            raise ValueError("No activation keys resolved for probe sweep.")

        shared_train_idx: list[int] | None = None
        shared_val_idx: list[int] | None = None
        trained: dict[str, TrainedLayerProbe] = {}
        for key in activation_keys:
            dataset = ProbingDataset.from_extraction_result(
                extraction,
                activation_key=key,
                labels=labels,
                positive_indices=positive_indices,
            )
            if shared_train_idx is None or shared_val_idx is None:
                shared_train_idx, shared_val_idx = self._build_split_indices(len(dataset))
            train_idx, val_idx = shared_train_idx, shared_val_idx
            train_loader, val_loader = self._build_loaders(dataset, train_idx, val_idx)
            trainer = BinaryLinearProbeTrainer(
                input_dim=dataset[0][0].numel(), config=self.config.probe
            )
            history = trainer.fit(train_loader, val_loader=val_loader)
            metrics = trainer.evaluate(val_loader)
            direction = self._normalized_direction(trainer)
            bias = float(trainer.model.linear.bias.detach().cpu().item())
            trained[key] = TrainedLayerProbe(
                activation_key=key,
                trainer=trainer,
                history=history,
                metrics=metrics,
                direction=direction,
                bias=bias,
            )
        return trained

    def _resolve_activation_keys(
        self, extraction: ExtractionResult | dict[str, Any]
    ) -> list[str]:
        if self.config.activation_targets is not None:
            resolved: list[str] = []
            for target in self.config.activation_targets:
                if isinstance(target, int):
                    resolved.append(f"layers_output:{target}")
                    continue
                key = target.strip()
                if ":" in key:
                    resolved.append(key)
                    continue
                try:
                    layer = int(key)
                except ValueError:
                    resolved.append(key)
                else:
                    resolved.append(f"layers_output:{layer}")
            return resolved
        requested = extraction.get("requested")
        if isinstance(requested, list) and requested:
            return list(requested)
        activations = extraction.get("activations", {})
        if isinstance(activations, dict):
            return list(activations.keys())
        return []

    def _build_split_indices(self, total_samples: int) -> tuple[list[int], list[int]]:
        if total_samples < 2:
            raise ValueError("Need at least 2 samples for train/validation splitting.")
        train_size = int(total_samples * self.config.train_fraction)
        train_size = min(max(train_size, 1), total_samples - 1)
        generator = None
        if self.config.split_seed is not None:
            generator = torch.Generator().manual_seed(int(self.config.split_seed))
        permutation = torch.randperm(total_samples, generator=generator).tolist()
        train_idx = permutation[:train_size]
        val_idx = permutation[train_size:]
        return train_idx, val_idx

    def _build_loaders(
        self, dataset: ProbingDataset, train_idx: list[int], val_idx: list[int]
    ) -> tuple[
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ]:
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        return train_loader, val_loader

    @staticmethod
    def _normalized_direction(trainer: BinaryLinearProbeTrainer) -> torch.Tensor:
        weight = trainer.model.linear.weight.detach().cpu().reshape(-1).float()
        norm = float(torch.linalg.vector_norm(weight).item())
        if norm == 0.0:
            return weight
        return weight / norm
