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
