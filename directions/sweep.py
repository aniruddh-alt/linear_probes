"""Layer-wise diff-in-means sweep orchestrator."""
from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import torch

from activation.types import ExtractionResult
from core.configs import SweepParams
from dataset import ProbingDataset
from dataset.splitting import _validate_split_indices
from directions.diff_means import DiffMeansEstimator, evaluate_projection
from directions.types import DiffMeansLayerResult, DiffMeansSweepResult
from probes.run_manifest import compute_dataset_fingerprint, write_run_manifest


class DiffMeansSweepRunner:
    """Runs diff-in-means direction finding across multiple activation layers."""

    def __init__(self, sweep: SweepParams | None = None):
        self.sweep = sweep or SweepParams()

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
    ) -> DiffMeansSweepResult:
        activation_keys = self._resolve_activation_keys(extraction)
        if not activation_keys:
            raise ValueError("No activation keys resolved for diff-means sweep.")

        sample_ids = extraction.get("sample_ids")
        if sample_ids is not None:
            normalized_sample_ids = [str(sid) for sid in sample_ids]
        else:
            total = self._resolve_total_samples(extraction)
            normalized_sample_ids = [str(i) for i in range(total)]

        estimator = DiffMeansEstimator()
        layer_results: dict[str, DiffMeansLayerResult] = {}
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

            # Fit on train split only
            train_features = dataset.features[train_indices]
            train_labels_t = dataset.labels[train_indices]
            layer_result = estimator.fit(train_features, train_labels_t, key=key)

            # Evaluate on val split
            val_features = dataset.features[val_indices]
            val_labels_t = dataset.labels[val_indices]
            val_metrics = evaluate_projection(val_features, val_labels_t, layer_result.direction)
            layer_result.val_metrics = val_metrics
            layer_results[key] = layer_result

        # Select best layer
        best_key, best_score = self._select_best_layer(layer_results)
        best_layer = layer_results[best_key]

        # Evaluate best on test split
        best_dataset = ProbingDataset.from_extraction_result(
            extraction, activation_key=best_key,
            labels=labels, positive_indices=positive_indices,
        )
        test_features = best_dataset.features[test_indices]
        test_labels_t = best_dataset.labels[test_indices]
        test_metrics = evaluate_projection(test_features, test_labels_t, best_layer.direction)

        # Controls: shuffled labels
        train_features = best_dataset.features[train_indices]
        train_labels_tensor = best_dataset.labels[train_indices]
        controls_summary = self._run_controls(
            estimator, train_features, train_labels_tensor,
            test_features, test_labels_t, best_layer.direction,
        )
        if self.sweep.enforce_control_sanity:
            self._enforce_control_sanity(controls_summary)

        # Fingerprint
        if dataset_labels is None:
            raise ValueError("Unable to resolve dataset labels.")
        dataset_fingerprint = compute_dataset_fingerprint(
            sample_ids=normalized_sample_ids,
            labels=dataset_labels,
            group_ids=group_ids,
        )

        # Manifest
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
                config={"method": "diff_means", "sweep": self.sweep},
                dataset_fingerprint=dataset_fingerprint,
                selected_key=best_key,
                selection_metric=self.sweep.selection_metric,
                split_indices={
                    "train": train_indices, "val": val_indices, "test": test_indices,
                },
                split_sizes=(len(train_indices), len(val_indices), len(test_indices)),
                test_metrics=test_metrics,
                controls=controls_summary,
            )

        return DiffMeansSweepResult(
            layers=layer_results,
            best_key=best_key,
            best_metric=self.sweep.selection_metric,
            best_score=best_score,
            best_direction=best_layer.direction,
            test_metrics=test_metrics,
            controls=controls_summary,
            split_sizes=(len(train_indices), len(val_indices), len(test_indices)),
            dataset_fingerprint=dataset_fingerprint,
            manifest_path=resolved_manifest_path,
        )

    def _run_controls(
        self,
        estimator: DiffMeansEstimator,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        real_direction: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        real_metrics = evaluate_projection(test_features, test_labels, real_direction)

        shuffled_runs: list[dict[str, float]] = []
        for seed in self.sweep.control_seeds:
            generator = torch.Generator().manual_seed(seed + 1_000)
            perm = torch.randperm(len(train_labels), generator=generator)
            shuffled_labels = train_labels[perm]
            shuffled_result = estimator.fit(train_features, shuffled_labels)
            shuffled_metrics = evaluate_projection(
                test_features, test_labels, shuffled_result.direction
            )
            shuffled_runs.append(shuffled_metrics)

        return {
            "real": _aggregate_metrics_list([real_metrics]),
            "shuffled_labels": _aggregate_metrics_list(shuffled_runs),
        }

    def _select_best_layer(
        self, results: dict[str, DiffMeansLayerResult]
    ) -> tuple[str, float]:
        if not results:
            raise ValueError("No layer results for selection.")
        chooser = max if self.sweep.maximize_metric else min
        best_key = chooser(
            results,
            key=lambda k: self._metric_value(
                results[k].val_metrics, self.sweep.selection_metric
            ),
        )
        return best_key, self._metric_value(
            results[best_key].val_metrics, self.sweep.selection_metric
        )

    @staticmethod
    def _metric_value(metrics: dict[str, float | tuple[float, float]], name: str) -> float:
        value = metrics.get(name) or metrics.get(name.lower()) or metrics.get(name.upper())
        if value is None or isinstance(value, tuple):
            raise KeyError(f"Metric '{name}' missing or non-scalar.")
        return float(value)

    def _resolve_activation_keys(
        self, extraction: ExtractionResult | dict[str, Any]
    ) -> list[str]:
        if self.sweep.activation_targets is not None:
            resolved: list[str] = []
            for target in self.sweep.activation_targets:
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

    @staticmethod
    def _resolve_total_samples(extraction: ExtractionResult | dict[str, Any]) -> int:
        activations = extraction.get("activations", {})
        if not isinstance(activations, dict) or not activations:
            return 0
        first_value = next(iter(activations.values()))
        if isinstance(first_value, torch.Tensor):
            return int(first_value.shape[0]) if first_value.ndim > 0 else 1
        return 0

    @staticmethod
    def _next_manifest_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem, suffix, parent = path.stem, path.suffix, path.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _enforce_control_sanity(self, controls: dict[str, dict[str, float]]) -> None:
        real_auroc = float(controls.get("real", {}).get("auroc_mean", 0.0))
        shuffled_auroc = float(controls.get("shuffled_labels", {}).get("auroc_mean", 0.0))
        if real_auroc <= shuffled_auroc:
            raise ValueError(
                "Control sanity check failed: real AUROC does not exceed shuffled labels."
            )


def _aggregate_metrics_list(
    runs: Sequence[dict[str, float]],
) -> dict[str, float]:
    if not runs:
        return {}
    grouped: dict[str, list[float]] = {}
    for run in runs:
        for key, value in run.items():
            if isinstance(value, (int, float)):
                grouped.setdefault(key, []).append(float(value))
    summary: dict[str, float] = {}
    for key, values in grouped.items():
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary
