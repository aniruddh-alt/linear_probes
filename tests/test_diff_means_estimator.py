from __future__ import annotations

import unittest

import torch

from sonde.directions.diff_means import DiffMeansEstimator, evaluate_projection


class DiffMeansEstimatorTests(unittest.TestCase):
    def test_computes_normalized_direction(self) -> None:
        torch.manual_seed(42)
        n = 100
        pos_features = torch.randn(n // 2, 4) + torch.tensor([2.0, 0, 0, 0])
        neg_features = torch.randn(n // 2, 4) + torch.tensor([-2.0, 0, 0, 0])
        features = torch.cat([pos_features, neg_features], dim=0)
        labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()

        estimator = DiffMeansEstimator()
        result = estimator.fit(features, labels)

        self.assertEqual(result.direction.shape, (4,))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(result.direction)), 1.0, places=5
        )
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
        torch.manual_seed(0)
        n = 60
        pos = torch.randn(n // 2, 3) + torch.tensor([3.0, 0, 0])
        neg = torch.randn(n // 2, 3) + torch.tensor([-3.0, 0, 0])
        features = torch.cat([pos, neg])
        labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()

        estimator = DiffMeansEstimator()
        result = estimator.fit(features, labels)
        scores = estimator.score(features, result.direction)

        pos_scores = scores[: n // 2]
        neg_scores = scores[n // 2 :]
        self.assertGreater(float(pos_scores.mean()), float(neg_scores.mean()))


class EvaluateProjectionTests(unittest.TestCase):
    def test_perfect_separation_gives_high_auroc(self) -> None:
        torch.manual_seed(0)
        n = 100
        direction = torch.tensor([1.0, 0.0, 0.0])
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


if __name__ == "__main__":
    unittest.main()
