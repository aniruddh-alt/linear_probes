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
        informative = torch.stack(
            [
                z + 0.05 * torch.randn(n),
                0.1 * torch.randn(n),
                0.1 * torch.randn(n),
            ],
            dim=1,
        )
        noisy = torch.randn(n, 3)
        layers = (
            {0: noisy, 1: informative}
            if informative_layer == 1
            else {0: informative, 1: noisy}
        )
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
        extraction, _labels, train_idx, val_idx, test_idx = self._make_extraction(180)
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
        extraction, _labels, train_idx, val_idx, test_idx = self._make_extraction(180)
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
        extraction, _labels, train_idx, val_idx, test_idx = self._make_extraction(90)
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
