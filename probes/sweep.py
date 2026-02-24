"""High-level orchestrator for layerwise linear probe sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from activation.types import ExtractionResult
from configs import LayerProbeSweepConfig
from dataset import ProbingDataset
from dataset.splitting import _validate_split_indices
from probes.linear import BinaryLinearProbeTrainer, run_probe_with_controls
from probes.run_manifest import compute_dataset_fingerprint, write_run_manifest
from probes.types import LayerProbeSweepResult, TrainedLayerProbe


class LayerProbeSweepRunner:
    """Runs train/validation split + fit/eval for multiple activation layers."""

    def __init__(self, config: LayerProbeSweepConfig | None = None):
        self.config = config or LayerProbeSweepConfig()

    def run(
        self,
        extraction: ExtractionResult | dict[str, Any],
        *,
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
        labels: list[int] | None = None,
        positive_indices: list[int] | None = None,
        group_ids: list[str] | None = None,
        manifest_path: str | Path | None = None,
        manifest_overwrite: bool = False,
        manifest_unique_path: bool = False,
    ) -> LayerProbeSweepResult:
        activation_keys = self._resolve_activation_keys(extraction)
        if not activation_keys:
            raise ValueError("No activation keys resolved for probe sweep.")

        trained: dict[str, TrainedLayerProbe] = {}
        sample_ids = extraction.get("sample_ids")
        if sample_ids is not None:
            normalized_sample_ids = [str(sample_id) for sample_id in sample_ids]
        else:
            total = self._resolve_total_samples(extraction)
            normalized_sample_ids = [str(i) for i in range(total)]

        dataset_labels: list[int] | None = None
        for key in activation_keys:
            dataset = ProbingDataset.from_extraction_result(
                extraction,
                activation_key=key,
                labels=labels,
                positive_indices=positive_indices,
            )
            current_labels = [int(label) for label in dataset.labels.tolist()]
            if dataset_labels is None:
                dataset_labels = current_labels
                _validate_split_indices(
                    total_samples=len(dataset),
                    labels=dataset_labels,
                    train_idx=train_indices,
                    val_idx=val_indices,
                    test_idx=test_indices,
                    group_ids=group_ids,
                )
            elif current_labels != dataset_labels:
                raise ValueError("Labels must be identical across activation keys.")

            train_loader, val_loader, test_loader = self._build_loaders(
                dataset, train_indices, val_indices, test_indices
            )
            trainer = BinaryLinearProbeTrainer(
                input_dim=dataset[0][0].numel(), config=self.config.probe
            )
            history = trainer.fit(train_loader, val_loader=val_loader)
            val_metrics = trainer.evaluate(val_loader)
            direction = self._normalized_direction(trainer)
            bias = float(trainer.model.linear.bias.detach().cpu().item())
            trained[key] = TrainedLayerProbe(
                activation_key=key,
                trainer=trainer,
                history=history,
                val_metrics=val_metrics,
                direction=direction,
                bias=bias,
            )

        best_key, best_score = self._select_best_layer(trained)
        best_probe = trained[best_key]
        first_dataset = ProbingDataset.from_extraction_result(
            extraction,
            activation_key=best_key,
            labels=labels,
            positive_indices=positive_indices,
        )
        train_loader, _, test_loader = self._build_loaders(
            first_dataset, train_indices, val_indices, test_indices
        )
        test_metrics = best_probe.trainer.evaluate(test_loader)
        controls_result = run_probe_with_controls(
            input_dim=first_dataset[0][0].numel(),
            train_loader=train_loader,
            eval_loader=test_loader,
            config=self.config.probe,
            seeds=self.config.control_seeds,
        )
        controls_summary = {
            "real": controls_result["real"],
            "shuffled_labels": controls_result["controls"]["shuffled_labels"],
            "random_features": controls_result["controls"]["random_features"],
        }
        if self.config.enforce_control_sanity:
            self._enforce_control_sanity(controls_summary)

        if dataset_labels is None:
            raise ValueError("Unable to resolve dataset labels for sweep output.")
        dataset_fingerprint = compute_dataset_fingerprint(
            sample_ids=normalized_sample_ids,
            labels=dataset_labels,
            group_ids=group_ids,
        )
        resolved_manifest_path: str | None = None
        if manifest_path is not None:
            manifest_path_obj = Path(manifest_path)
            if manifest_overwrite and manifest_unique_path:
                raise ValueError(
                    "manifest_overwrite and manifest_unique_path are mutually exclusive."
                )
            if manifest_overwrite and manifest_path_obj.exists():
                manifest_path_obj.unlink()
            if manifest_unique_path:
                manifest_path_obj = self._next_manifest_path(manifest_path_obj)
            resolved_manifest_path = write_run_manifest(
                manifest_path=manifest_path_obj,
                config=self.config,
                dataset_fingerprint=dataset_fingerprint,
                selected_key=best_key,
                selection_metric=self.config.selection_metric,
                split_indices={
                    "train": train_indices,
                    "val": val_indices,
                    "test": test_indices,
                },
                split_sizes=(len(train_indices), len(val_indices), len(test_indices)),
                test_metrics=test_metrics,
                controls=controls_summary,
            )

        return LayerProbeSweepResult(
            probes=trained,
            best_key=best_key,
            best_metric=self.config.selection_metric,
            best_score=best_score,
            test_metrics=test_metrics,
            controls=controls_summary,
            best_direction=best_probe.direction,
            best_bias=best_probe.bias,
            split_sizes=(len(train_indices), len(val_indices), len(test_indices)),
            dataset_fingerprint=dataset_fingerprint,
            manifest_path=resolved_manifest_path,
        )

    @staticmethod
    def _next_manifest_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

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

    def _build_loaders(
        self,
        dataset: ProbingDataset,
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
    ) -> tuple[
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ]:
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        return train_loader, val_loader, test_loader

    @staticmethod
    def _normalized_direction(trainer: BinaryLinearProbeTrainer) -> torch.Tensor:
        weight = trainer.model.linear.weight.detach().cpu().reshape(-1).float()
        norm = float(torch.linalg.vector_norm(weight).item())
        if norm == 0.0:
            return weight
        return weight / norm

    def _select_best_layer(
        self, probes: dict[str, TrainedLayerProbe]
    ) -> tuple[str, float]:
        if not probes:
            raise ValueError("No probes available for layer selection.")
        chooser = max if self.config.maximize_metric else min
        best_key = chooser(
            probes,
            key=lambda key: self._metric_value(
                probes[key].val_metrics, self.config.selection_metric
            ),
        )
        best_score = self._metric_value(
            probes[best_key].val_metrics, self.config.selection_metric
        )
        return best_key, best_score

    @staticmethod
    def _metric_value(metrics: dict[str, float | tuple[float, float]], name: str) -> float:
        value = metrics.get(name)
        if value is None:
            value = metrics.get(name.lower())
        if value is None:
            value = metrics.get(name.upper())
        if value is None or isinstance(value, tuple):
            raise KeyError(f"Metric '{name}' missing or non-scalar.")
        return float(value)

    @staticmethod
    def _resolve_total_samples(extraction: ExtractionResult | dict[str, Any]) -> int:
        activations = extraction.get("activations", {})
        if not isinstance(activations, dict) or not activations:
            return 0
        first_value = next(iter(activations.values()))
        if isinstance(first_value, torch.Tensor):
            return int(first_value.shape[0]) if first_value.ndim > 0 else 1
        return 0

    def _enforce_control_sanity(self, controls: dict[str, dict[str, float]]) -> None:
        real = controls.get("real", {})
        shuffled = controls.get("shuffled_labels", {})
        random_features = controls.get("random_features", {})
        real_auroc = float(real.get("auroc_mean", 0.0))
        shuffled_auroc = float(shuffled.get("auroc_mean", 0.0))
        random_auroc = float(random_features.get("auroc_mean", 0.0))
        if real_auroc <= max(shuffled_auroc, random_auroc):
            raise ValueError(
                "Control sanity check failed: real AUROC does not exceed controls."
            )
