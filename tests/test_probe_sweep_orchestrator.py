from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from core.configs import ProbeParams, SweepParams
from dataset import ProbingSampleBuilder
from probes import LayerProbeSweepRunner


class LayerProbeSweepRunnerTests(unittest.TestCase):
    @staticmethod
    def _split_from_labels(labels: list[int], sample_ids: list[str]):
        records = [
            {"id": sample_ids[idx], "text": f"sample-{idx}", "label": labels[idx]}
            for idx in range(len(labels))
        ]
        bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
        return bundle.train_val_test_split(group_ids=bundle.ids, seed=123)

    def test_runner_trains_all_layers_and_selects_best(self) -> None:
        torch.manual_seed(0)
        n = 180
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
        train_idx, val_idx, test_idx = self._split_from_labels(
            labels=labels,
            sample_ids=extraction["sample_ids"],
        )

        result = LayerProbeSweepRunner(
            probe=ProbeParams(epochs=20, learning_rate=0.05, seed=7, weight_decay=0.01),
            sweep=SweepParams(activation_targets=[0, 1], batch_size=32, selection_metric="auroc"),
        ).run(
            extraction,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            group_ids=extraction["sample_ids"],
        )
        probes = result.probes

        self.assertEqual(set(probes.keys()), {"layers_output:0", "layers_output:1"})
        self.assertEqual(result.best_key, "layers_output:1")
        self.assertGreater(float(result.best_score), 0.9)
        self.assertEqual(int(probes[result.best_key].direction.ndim), 1)
        self.assertIn("auroc", result.test_metrics)
        self.assertIn("real", result.controls)
        self.assertIn("shuffled_labels", result.controls)
        self.assertIn("random_features", result.controls)
        self.assertEqual(result.split_sizes, (len(train_idx), len(val_idx), len(test_idx)))

    def test_runner_writes_manifest_and_test_only_after_selection(self) -> None:
        torch.manual_seed(0)
        n = 90
        features = torch.randn(n, 2)
        labels = (features[:, 0] > 0).long().tolist()
        extraction = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": features},
            "sample_ids": [f"id-{i}" for i in range(n)],
            "labels": labels,
        }
        records = [
            {"id": extraction["sample_ids"][i], "text": f"sample-{i}", "label": labels[i]}
            for i in range(len(labels))
        ]
        bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
        train_idx, val_idx, test_idx = bundle.train_val_test_split(
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=1,
            group_ids=bundle.ids,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "run_manifest.json"
            result = LayerProbeSweepRunner(
                probe=ProbeParams(epochs=10, learning_rate=0.1, seed=1),
                sweep=SweepParams(activation_targets=[0], batch_size=8),
            ).run(
                extraction,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                manifest_path=manifest,
            )
        self.assertIn("layers_output:0", result.probes)
        self.assertGreaterEqual(
            float(result.probes["layers_output:0"].val_metrics["accuracy"]), 0.6
        )
        self.assertTrue(isinstance(result.dataset_fingerprint, str))
        self.assertEqual(result.manifest_path, str(manifest))

    def test_runner_overwrites_manifest_when_requested(self) -> None:
        torch.manual_seed(0)
        n = 60
        features = torch.randn(n, 2)
        labels = (features[:, 0] > 0).long().tolist()
        extraction = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": features},
            "sample_ids": [f"id-{i}" for i in range(n)],
            "labels": labels,
        }
        records = [
            {"id": extraction["sample_ids"][i], "text": f"sample-{i}", "label": labels[i]}
            for i in range(len(labels))
        ]
        bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
        train_idx, val_idx, test_idx = bundle.train_val_test_split(seed=9, group_ids=bundle.ids)
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "run_manifest.json"
            manifest.write_text("old", encoding="utf-8")
            result = LayerProbeSweepRunner(
                probe=ProbeParams(epochs=5, learning_rate=0.05, seed=9),
                sweep=SweepParams(activation_targets=["layers_output:0"], batch_size=8),
            ).run(
                extraction,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                manifest_path=manifest,
                manifest_overwrite=True,
            )
            self.assertEqual(result.manifest_path, str(manifest))
            self.assertNotEqual(manifest.read_text(encoding="utf-8"), "old")

    def test_runner_uses_unique_manifest_path_when_requested(self) -> None:
        torch.manual_seed(0)
        n = 60
        features = torch.randn(n, 2)
        labels = (features[:, 0] > 0).long().tolist()
        extraction = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": features},
            "sample_ids": [f"id-{i}" for i in range(n)],
            "labels": labels,
        }
        records = [
            {"id": extraction["sample_ids"][i], "text": f"sample-{i}", "label": labels[i]}
            for i in range(len(labels))
        ]
        bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
        train_idx, val_idx, test_idx = bundle.train_val_test_split(seed=7, group_ids=bundle.ids)
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "run_manifest.json"
            manifest.write_text("already here", encoding="utf-8")
            result = LayerProbeSweepRunner(
                probe=ProbeParams(epochs=5, learning_rate=0.05, seed=7),
                sweep=SweepParams(activation_targets=["layers_output:0"], batch_size=8),
            ).run(
                extraction,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                manifest_path=manifest,
                manifest_unique_path=True,
            )
            self.assertEqual(manifest.read_text(encoding="utf-8"), "already here")
            self.assertIsNotNone(result.manifest_path)
            assert result.manifest_path is not None
            self.assertTrue(result.manifest_path.endswith("run_manifest_1.json"))

    def test_runner_rejects_mutually_exclusive_manifest_modes(self) -> None:
        torch.manual_seed(0)
        n = 30
        features = torch.randn(n, 2)
        labels = (features[:, 0] > 0).long().tolist()
        extraction = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": features},
            "sample_ids": [f"id-{i}" for i in range(n)],
            "labels": labels,
        }
        records = [
            {"id": extraction["sample_ids"][i], "text": f"sample-{i}", "label": labels[i]}
            for i in range(len(labels))
        ]
        bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
        train_idx, val_idx, test_idx = bundle.train_val_test_split(seed=2, group_ids=bundle.ids)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LayerProbeSweepRunner().run(
                extraction,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                manifest_path=Path("unused.json"),
                manifest_overwrite=True,
                manifest_unique_path=True,
            )


if __name__ == "__main__":
    unittest.main()

    def test_sweep_runner_with_mean_probe(self) -> None:
        torch.manual_seed(0)
        n = 120
        z = torch.randn(n)
        labels = (z > 0).long().tolist()
        features_l0 = [torch.randn(5, 3) for _ in range(n)]
        features_l1 = [
            torch.stack([z[i].expand(3) + 0.05 * torch.randn(3)] * 5)
            for i in range(n)
        ]
        extraction = {
            "requested": ["layers_output:0", "layers_output:1"],
            "activations": {
                "layers_output:0": features_l0,
                "layers_output:1": features_l1,
            },
            "sample_ids": [f"id-{i}" for i in range(n)],
            "labels": labels,
        }
        train_idx, val_idx, test_idx = self._split_from_labels(
            labels=labels, sample_ids=extraction["sample_ids"]
        )
        result = LayerProbeSweepRunner(
            probe=ProbeParams(
                probe_type="mean", epochs=15, learning_rate=0.05,
                seed=7, weight_decay=0.01, early_stopping_patience=None,
            ),
            sweep=SweepParams(
                activation_targets=[0, 1], batch_size=16, selection_metric="auroc"
            ),
        ).run(
            extraction,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )
        self.assertIsNotNone(result.test_metrics)
        self.assertIn("auroc", result.test_metrics)
