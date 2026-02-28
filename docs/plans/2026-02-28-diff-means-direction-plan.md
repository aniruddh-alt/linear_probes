# Diff-in-Means Direction Finding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add diff-in-means as a direction-finding method that computes `r_l = mean(H+) - mean(H-)` per layer, with layer sweep, evaluation, controls, and config-driven execution — all parallel to the existing probe infrastructure.

**Architecture:** New `directions/` module with `DiffMeansEstimator` (core computation), `DiffMeansSweepRunner` (layer sweep + eval), result dataclasses, and `DiffMeansConfig` for YAML-driven execution. Evaluation via projection scoring (dot product with direction → AUROC). Reuses existing splitting, dataset, manifest, and activation infrastructure.

**Tech Stack:** PyTorch, torchmetrics (BinaryAUROC), OmegaConf configs, existing `ProbingDataset` / `ExtractionResult` / `run_manifest` infrastructure.

---

### Task 1: DiffMeansLayerResult and DiffMeansSweepResult types

**Files:**
- Create: `directions/types.py`
- Test: `tests/test_diff_means_types.py`

**Step 1: Write the failing test**

```python
# tests/test_diff_means_types.py
from __future__ import annotations
import unittest
import torch
from directions.types import DiffMeansLayerResult, DiffMeansSweepResult


class DiffMeansTypesTests(unittest.TestCase):
    def test_layer_result_construction(self) -> None:
        direction = torch.randn(64)
        result = DiffMeansLayerResult(
            key="layers_output:5",
            direction=direction,
            raw_norm=1.23,
            positive_count=50,
            negative_count=50,
            val_metrics={"auroc": 0.85, "accuracy": 0.80},
        )
        self.assertEqual(result.key, "layers_output:5")
        self.assertAlmostEqual(result.raw_norm, 1.23, places=2)
        self.assertEqual(result.positive_count, 50)

    def test_sweep_result_construction(self) -> None:
        direction = torch.randn(64)
        layer_result = DiffMeansLayerResult(
            key="layers_output:5",
            direction=direction,
            raw_norm=1.0,
            positive_count=50,
            negative_count=50,
            val_metrics={"auroc": 0.9},
        )
        result = DiffMeansSweepResult(
            layers={"layers_output:5": layer_result},
            best_key="layers_output:5",
            best_metric="auroc",
            best_score=0.9,
            best_direction=direction,
            test_metrics={"auroc": 0.88},
            controls={"real": {"auroc_mean": 0.9}, "shuffled_labels": {"auroc_mean": 0.52}},
            split_sizes=(100, 30, 30),
            dataset_fingerprint="abc123",
            manifest_path=None,
        )
        self.assertEqual(result.best_key, "layers_output:5")
        self.assertIsNone(result.manifest_path)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diff_means_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'directions'`

**Step 3: Write minimal implementation**

```python
# directions/__init__.py
```

```python
# directions/types.py
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diff_means_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add directions/__init__.py directions/types.py tests/test_diff_means_types.py
git commit -m "feat: add DiffMeansLayerResult and DiffMeansSweepResult types"
```

---

### Task 2: DiffMeansEstimator — core computation

**Files:**
- Create: `directions/diff_means.py`
- Test: `tests/test_diff_means_estimator.py`

**Step 1: Write the failing test**

```python
# tests/test_diff_means_estimator.py
from __future__ import annotations
import unittest
import torch
from directions.diff_means import DiffMeansEstimator


class DiffMeansEstimatorTests(unittest.TestCase):
    def test_computes_normalized_direction(self) -> None:
        torch.manual_seed(42)
        n = 100
        # Positive class shifted +2 in dim 0, negative class shifted -2
        pos_features = torch.randn(n // 2, 4) + torch.tensor([2.0, 0, 0, 0])
        neg_features = torch.randn(n // 2, 4) + torch.tensor([-2.0, 0, 0, 0])
        features = torch.cat([pos_features, neg_features], dim=0)
        labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()

        estimator = DiffMeansEstimator()
        result = estimator.fit(features, labels)

        # Direction should be roughly [1, 0, 0, 0] (normalized)
        self.assertEqual(result.direction.shape, (4,))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(result.direction)), 1.0, places=5)
        # First component should dominate
        self.assertGreater(abs(float(result.direction[0])), 0.9)
        self.assertEqual(result.positive_count, 50)
        self.assertEqual(result.negative_count, 50)
        self.assertGreater(result.raw_norm, 0.0)

    def test_raises_on_single_class(self) -> None:
        features = torch.randn(10, 4)
        labels = torch.ones(10).long()
        estimator = DiffMeansEstimator()
        with self.assertRaisesRegex(ValueError, "both classes"):
            estimator.fit(features, labels)

    def test_score_projects_correctly(self) -> None:
        """Projection scores should separate the classes."""
        torch.manual_seed(0)
        n = 60
        pos = torch.randn(n // 2, 3) + torch.tensor([3.0, 0, 0])
        neg = torch.randn(n // 2, 3) + torch.tensor([-3.0, 0, 0])
        features = torch.cat([pos, neg])
        labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()

        estimator = DiffMeansEstimator()
        result = estimator.fit(features, labels)
        scores = estimator.score(features, result.direction)

        # Positive class should have higher scores
        pos_scores = scores[:n // 2]
        neg_scores = scores[n // 2:]
        self.assertGreater(float(pos_scores.mean()), float(neg_scores.mean()))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diff_means_estimator.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# directions/diff_means.py
"""Diff-in-means direction estimator."""
from __future__ import annotations

import torch

from directions.types import DiffMeansLayerResult


class DiffMeansEstimator:
    """Computes the diff-in-means direction between two classes.

    Given activations and binary labels, computes:
        r = mean(H+) - mean(H-)
        r_hat = r / ||r||
    """

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        key: str = "",
        val_metrics: dict[str, float | tuple[float, float]] | None = None,
    ) -> DiffMeansLayerResult:
        labels = labels.long()
        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"Need both classes for diff-in-means (got {n_pos} positive, {n_neg} negative)."
            )
        pos_mean = features[pos_mask].float().mean(dim=0)
        neg_mean = features[neg_mask].float().mean(dim=0)
        raw_direction = pos_mean - neg_mean
        raw_norm = float(torch.linalg.vector_norm(raw_direction).item())
        if raw_norm == 0.0:
            direction = raw_direction
        else:
            direction = raw_direction / raw_norm
        return DiffMeansLayerResult(
            key=key,
            direction=direction,
            raw_norm=raw_norm,
            positive_count=n_pos,
            negative_count=n_neg,
            val_metrics=val_metrics or {},
        )

    @staticmethod
    def score(features: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Project features onto the direction vector. Returns 1D scores."""
        return features.float() @ direction.float()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diff_means_estimator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add directions/diff_means.py tests/test_diff_means_estimator.py
git commit -m "feat: add DiffMeansEstimator core computation"
```

---

### Task 3: Projection-based evaluation helper

**Files:**
- Modify: `directions/diff_means.py` (add `evaluate_projection`)
- Test: `tests/test_diff_means_estimator.py` (add evaluation tests)

**Step 1: Write the failing test**

Add to `tests/test_diff_means_estimator.py`:

```python
from directions.diff_means import DiffMeansEstimator, evaluate_projection

class EvaluateProjectionTests(unittest.TestCase):
    def test_perfect_separation_gives_high_auroc(self) -> None:
        torch.manual_seed(0)
        n = 100
        direction = torch.tensor([1.0, 0.0, 0.0])
        # Perfect separation along dim 0
        pos = torch.tensor([[5.0, 0, 0]] * (n // 2))
        neg = torch.tensor([[-5.0, 0, 0]] * (n // 2))
        features = torch.cat([pos, neg])
        labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()

        metrics = evaluate_projection(features, labels, direction)
        self.assertGreaterEqual(metrics["auroc"], 0.99)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)

    def test_random_direction_gives_chance_auroc(self) -> None:
        torch.manual_seed(1)
        n = 200
        features = torch.randn(n, 10)
        labels = torch.randint(0, 2, (n,))
        direction = torch.randn(10)
        direction = direction / torch.linalg.vector_norm(direction)

        metrics = evaluate_projection(features, labels, direction)
        self.assertGreater(metrics["auroc"], 0.3)
        self.assertLess(metrics["auroc"], 0.7)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diff_means_estimator.py::EvaluateProjectionTests -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `directions/diff_means.py`:

```python
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


def evaluate_projection(
    features: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Evaluate classification by projecting features onto a direction vector.

    Scores = features @ direction. AUROC computed from raw scores.
    For accuracy/F1: if threshold is None, uses median of scores.
    """
    scores = features.float() @ direction.float()
    labels = labels.long()

    # AUROC from raw scores (normalized to [0,1] via sigmoid for torchmetrics)
    probs = torch.sigmoid(scores)
    auroc_metric = BinaryAUROC()
    auroc = float(auroc_metric(probs, labels).item())

    # Threshold-based metrics
    if threshold is None:
        threshold = float(scores.median().item())
    preds = (scores >= threshold).long()
    accuracy = float(BinaryAccuracy()(preds.float(), labels).item())
    precision = float(BinaryPrecision(zero_division=0)(preds.float(), labels).item())
    recall = float(BinaryRecall(zero_division=0)(preds.float(), labels).item())
    f1 = float(BinaryF1Score(zero_division=0)(preds.float(), labels).item())

    return {
        "auroc": auroc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diff_means_estimator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add directions/diff_means.py tests/test_diff_means_estimator.py
git commit -m "feat: add evaluate_projection for direction-based classification"
```

---

### Task 4: DiffMeansSweepRunner — layer sweep orchestrator

**Files:**
- Create: `directions/sweep.py`
- Test: `tests/test_diff_means_sweep.py`

**Step 1: Write the failing test**

```python
# tests/test_diff_means_sweep.py
from __future__ import annotations
import unittest
import torch
from core.configs import SweepParams
from dataset import ProbingSampleBuilder
from directions.sweep import DiffMeansSweepRunner


class DiffMeansSweepRunnerTests(unittest.TestCase):
    @staticmethod
    def _make_extraction(n: int, informative_layer: int = 1) -> tuple:
        """Create a 2-layer extraction with one informative and one noisy layer."""
        torch.manual_seed(0)
        z = torch.randn(n)
        labels = (z > 0).long().tolist()
        # Informative: class means separated along dim 0
        informative = torch.stack([
            z + 0.05 * torch.randn(n),
            0.1 * torch.randn(n),
            0.1 * torch.randn(n),
        ], dim=1)
        noisy = torch.randn(n, 3)
        layers = {0: noisy, 1: informative} if informative_layer == 1 else {0: informative, 1: noisy}
        extraction = {
            "requested": ["layers_output:0", "layers_output:1"],
            "activations": {f"layers_output:{k}": v for k, v in layers.items()},
            "sample_ids": [f"id-{i}" for i in range(n)],
            "labels": labels,
        }
        records = [
            {"id": f"id-{i}", "text": f"sample-{i}", "label": labels[i]}
            for i in range(n)
        ]
        bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
        train_idx, val_idx, test_idx = bundle.train_val_test_split(
            group_ids=bundle.ids, seed=123
        )
        return extraction, labels, train_idx, val_idx, test_idx

    def test_selects_informative_layer(self) -> None:
        extraction, labels, train_idx, val_idx, test_idx = self._make_extraction(180)
        result = DiffMeansSweepRunner(
            sweep=SweepParams(activation_targets=[0, 1], selection_metric="auroc"),
        ).run(
            extraction,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            group_ids=extraction["sample_ids"],
        )
        self.assertEqual(result.best_key, "layers_output:1")
        self.assertGreater(result.best_score, 0.8)
        self.assertEqual(len(result.layers), 2)
        self.assertIn("auroc", result.test_metrics)
        self.assertEqual(result.best_direction.ndim, 1)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(result.best_direction)), 1.0, places=5
        )

    def test_controls_included(self) -> None:
        extraction, labels, train_idx, val_idx, test_idx = self._make_extraction(180)
        result = DiffMeansSweepRunner(
            sweep=SweepParams(
                activation_targets=[0, 1],
                control_seeds=[0, 1, 2],
            ),
        ).run(
            extraction,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )
        self.assertIn("real", result.controls)
        self.assertIn("shuffled_labels", result.controls)
        self.assertIn("auroc_mean", result.controls["real"])
        self.assertIn("auroc_mean", result.controls["shuffled_labels"])

    def test_single_layer(self) -> None:
        extraction, labels, train_idx, val_idx, test_idx = self._make_extraction(90)
        result = DiffMeansSweepRunner(
            sweep=SweepParams(activation_targets=[1]),
        ).run(
            extraction,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )
        self.assertEqual(len(result.layers), 1)
        self.assertEqual(result.best_key, "layers_output:1")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diff_means_sweep.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# directions/sweep.py
"""Layer-wise diff-in-means sweep orchestrator."""
from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import torch
from torch.utils.data import Subset

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
            train_labels = dataset.labels[train_indices]
            layer_result = estimator.fit(
                train_features, train_labels, key=key
            )

            # Evaluate on val split
            val_features = dataset.features[val_indices]
            val_labels = dataset.labels[val_indices]
            val_metrics = evaluate_projection(
                val_features, val_labels, layer_result.direction
            )
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
        test_labels = best_dataset.labels[test_indices]
        test_metrics = evaluate_projection(
            test_features, test_labels, best_layer.direction
        )

        # Controls: shuffled labels
        train_features = best_dataset.features[train_indices]
        train_labels_tensor = best_dataset.labels[train_indices]
        controls_summary = self._run_controls(
            estimator, train_features, train_labels_tensor,
            test_features, test_labels, best_layer.direction,
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
        # Real metrics
        real_metrics = evaluate_projection(test_features, test_labels, real_direction)

        # Shuffled label controls
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diff_means_sweep.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add directions/sweep.py tests/test_diff_means_sweep.py
git commit -m "feat: add DiffMeansSweepRunner layer sweep orchestrator"
```

---

### Task 5: DiffMeansConfig and experiment runner integration

**Files:**
- Create: `core/configs/diff_means_config.py`
- Modify: `core/configs/__init__.py` — add DiffMeansConfig to exports
- Modify: `runners/experiment_runner.py` — add action handler
- Modify: `directions/__init__.py` — add public exports
- Test: `tests/test_diff_means_config.py`

**Step 1: Write the failing test**

```python
# tests/test_diff_means_config.py
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from core.configs.diff_means_config import DiffMeansConfig
from runners.experiment_runner import load_run_config, STAGE_CONFIG_MAP


class DiffMeansConfigTests(unittest.TestCase):
    def test_default_construction(self) -> None:
        cfg = DiffMeansConfig()
        self.assertEqual(cfg.action, "diff_means")
        self.assertEqual(cfg.seed, 0)

    def test_from_yaml(self) -> None:
        yaml_content = """\
action: diff_means
run_name: test-dm
seed: 42
split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
sweep:
  selection_metric: auroc
  activation_targets: [0, 1, 2]
io:
  input_path: data/activations_manifest.pt
  output_dir: artifacts
output:
  save_plots: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_run_config(f.name)
        self.assertIsInstance(cfg, DiffMeansConfig)
        self.assertEqual(cfg.run_name, "test-dm")
        self.assertEqual(cfg.seed, 42)

    def test_registered_in_stage_config_map(self) -> None:
        self.assertIn("diff_means", STAGE_CONFIG_MAP)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diff_means_config.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `core/configs/diff_means_config.py`:

```python
"""Config for the diff_means stage."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig
from core.configs.params.io_params import IOParams
from core.configs.params.output_params import OutputParams
from core.configs.params.split_params import SplitParams
from core.configs.params.sweep_params import SweepParams


@dataclass
class DiffMeansConfig(BaseConfig):
    """Configuration for the diff_means stage (action: diff_means)."""

    run_name: str = ""
    seed: int = 0
    action: str = "diff_means"
    split: SplitParams = field(default_factory=SplitParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    io: IOParams = field(default_factory=IOParams)
    output: OutputParams = field(default_factory=OutputParams)
```

Update `core/configs/__init__.py` — add:
```python
from core.configs.diff_means_config import DiffMeansConfig
```
and add `"DiffMeansConfig"` to `__all__`.

Update `runners/experiment_runner.py`:
- Import `DiffMeansConfig`
- Add `"diff_means": DiffMeansConfig` to `STAGE_CONFIG_MAP`
- Update `StageConfig` union type
- Add `dispatch_action` branch for `DiffMeansConfig`
- Add `_action_diff_means` stub (returns configured status for now)

```python
def _action_diff_means(cfg: DiffMeansConfig) -> RunResult:
    """Placeholder for diff-means action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "seed": cfg.seed,
            "status": "configured",
        },
    )
```

Update `directions/__init__.py`:
```python
from .diff_means import DiffMeansEstimator, evaluate_projection
from .sweep import DiffMeansSweepRunner
from .types import DiffMeansLayerResult, DiffMeansSweepResult

__all__ = [
    "DiffMeansEstimator",
    "DiffMeansSweepRunner",
    "DiffMeansLayerResult",
    "DiffMeansSweepResult",
    "evaluate_projection",
]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diff_means_config.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest --tb=short -q`
Expected: All tests pass (106 existing + new tests)

**Step 6: Commit**

```bash
git add core/configs/diff_means_config.py core/configs/__init__.py runners/experiment_runner.py directions/__init__.py tests/test_diff_means_config.py
git commit -m "feat: add DiffMeansConfig and experiment runner integration"
```

---

### Task 6: Verify full test suite and finalize

**Step 1: Run all tests**

Run: `python -m pytest --tb=short -q`
Expected: All tests pass

**Step 2: Verify imports work end-to-end**

Run: `python -c "from directions import DiffMeansEstimator, DiffMeansSweepRunner, evaluate_projection; print('OK')"`
Expected: `OK`

Run: `python -c "from core.configs import DiffMeansConfig; print(DiffMeansConfig().action)"`
Expected: `diff_means`

**Step 3: Final commit if any cleanup needed**
