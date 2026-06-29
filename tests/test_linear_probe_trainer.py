from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from sonde.core.configs import ProbeParams
from sonde.probes.architectures import build_probe
from sonde.probes.linear import (
    BinaryLinearProbeTrainer,
    BinaryProbeTrainer,
    run_probe_with_controls,
)


def _scalar(v: float | tuple[float, float]) -> float:
    return v if isinstance(v, (int, float)) else v[0]


class LinearProbeTrainerTests(unittest.TestCase):
    def test_trainer_learns_linearly_separable_problem(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(128, 4)
        labels = (features[:, 0] > 0).long()
        loader = DataLoader(
            TensorDataset(features, labels), batch_size=32, shuffle=True
        )

        trainer = BinaryLinearProbeTrainer(
            input_dim=4,
            config=ProbeParams(
                epochs=25, learning_rate=0.1, early_stopping_patience=None
            ),
        )
        history = trainer.fit(loader, val_loader=loader)
        metrics = trainer.evaluate(loader)

        self.assertEqual(len(history["train_loss"]), 25)
        self.assertGreater(_scalar(metrics["accuracy"]), 0.9)
        self.assertIn("f1", metrics)
        self.assertIn("auroc", metrics)

    def test_control_runner_returns_real_and_control_summaries(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(64, 3)
        labels = (features[:, 0] > 0).long()
        loader = DataLoader(
            TensorDataset(features, labels), batch_size=16, shuffle=True
        )

        result = run_probe_with_controls(
            input_dim=3,
            train_loader=loader,
            eval_loader=loader,
            config=ProbeParams(epochs=5, learning_rate=0.1),
            seeds=(0, 1),
        )

        self.assertIn("real", result)
        self.assertIn("controls", result)
        self.assertIn("accuracy_mean", result["real"])
        self.assertIn("shuffled_labels", result["controls"])
        self.assertIn("random_features", result["controls"])


class GenericProbeTrainerTests(unittest.TestCase):
    def test_trainer_accepts_any_module(self) -> None:
        model = build_probe("mean", input_dim=4)
        trainer = BinaryProbeTrainer(model=model, config=ProbeParams(epochs=5))
        self.assertIs(trainer.model, model)

    def test_trainer_trains_attention_probe(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(64, 10, 4)
        labels = (features[:, -1, 0] > 0).long()
        mask = torch.ones(64, 10)
        dataset = TensorDataset(features, labels, mask)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = build_probe("attention", input_dim=4)
        trainer = BinaryProbeTrainer(
            model=model,
            config=ProbeParams(
                epochs=20, learning_rate=0.01, early_stopping_patience=None
            ),
        )
        trainer.fit(loader, val_loader=loader)
        metrics = trainer.evaluate(loader)
        self.assertIn("auroc", metrics)

    def test_backward_compat_linear_trainer_still_works(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(64, 4)
        labels = (features[:, 0] > 0).long()
        loader = DataLoader(
            TensorDataset(features, labels), batch_size=16, shuffle=True
        )

        trainer = BinaryLinearProbeTrainer(
            input_dim=4,
            config=ProbeParams(
                epochs=10, learning_rate=0.1, early_stopping_patience=None
            ),
        )
        trainer.fit(loader, val_loader=loader)
        metrics = trainer.evaluate(loader)
        self.assertGreater(_scalar(metrics["accuracy"]), 0.8)


if __name__ == "__main__":
    unittest.main()
