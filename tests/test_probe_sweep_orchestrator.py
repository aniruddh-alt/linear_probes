from __future__ import annotations

import unittest

import torch

from configs import LayerProbeSweepConfig, ProbeConfig
from probes import LayerProbeSweepRunner


class LayerProbeSweepRunnerTests(unittest.TestCase):
    def test_runner_trains_all_layers_and_selects_best(self) -> None:
        torch.manual_seed(0)
        n = 160
        z = torch.randn(n)
        labels = (z > 0).long().tolist()

        informative = torch.stack(
            (
                z + 0.05 * torch.randn(n),
                0.1 * torch.randn(n),
                0.1 * torch.randn(n),
            ),
            dim=1,
        )
        noisy = torch.randn(n, 3)
        extraction = {
            "requested": ["layers_output:0", "layers_output:1"],
            "activations": {
                "layers_output:0": noisy,
                "layers_output:1": informative,
            },
            "sample_ids": [f"id-{i}" for i in range(n)],
            "labels": labels,
        }

        config = LayerProbeSweepConfig(
            activation_targets=[0, 1],
            train_fraction=0.8,
            batch_size=32,
            split_seed=123,
            selection_metric="accuracy",
            probe=ProbeConfig(epochs=20, learning_rate=0.05, seed=7),
        )
        probes = LayerProbeSweepRunner(config).run(extraction)

        self.assertEqual(set(probes.keys()), {"layers_output:0", "layers_output:1"})
        best_key = max(probes, key=lambda key: float(probes[key].metrics["accuracy"]))
        self.assertEqual(best_key, "layers_output:1")
        self.assertGreater(float(probes[best_key].metrics["accuracy"]), 0.9)
        self.assertEqual(int(probes[best_key].direction.ndim), 1)


if __name__ == "__main__":
    unittest.main()
