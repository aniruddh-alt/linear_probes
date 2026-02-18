from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from probes.linear import BinaryLinearProbeTrainer, ProbeTrainConfig, run_probe_with_controls


class LinearProbeTrainerTests(unittest.TestCase):
    def test_trainer_learns_linearly_separable_problem(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(128, 4)
        labels = (features[:, 0] > 0).long()
        loader = DataLoader(TensorDataset(features, labels), batch_size=32, shuffle=True)

        trainer = BinaryLinearProbeTrainer(
            input_dim=4,
            config=ProbeTrainConfig(epochs=25, learning_rate=0.1),
        )
        history = trainer.fit(loader, val_loader=loader)
        metrics = trainer.evaluate(loader)

        self.assertEqual(len(history["train_loss"]), 25)
        self.assertGreater(metrics["accuracy"], 0.9)
        self.assertIn("f1", metrics)
        self.assertIn("auroc", metrics)

    def test_control_runner_returns_real_and_control_summaries(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(64, 3)
        labels = (features[:, 0] > 0).long()
        loader = DataLoader(TensorDataset(features, labels), batch_size=16, shuffle=True)

        result = run_probe_with_controls(
            input_dim=3,
            train_loader=loader,
            eval_loader=loader,
            config=ProbeTrainConfig(epochs=5, learning_rate=0.1),
            seeds=(0, 1),
        )

        self.assertIn("real", result)
        self.assertIn("controls", result)
        self.assertIn("accuracy_mean", result["real"])
        self.assertIn("shuffled_labels", result["controls"])
        self.assertIn("random_features", result["controls"])


if __name__ == "__main__":
    unittest.main()
