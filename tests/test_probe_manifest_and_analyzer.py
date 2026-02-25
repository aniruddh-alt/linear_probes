from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from core.configs import ProbeParams, SweepParams
from probes.analyze import ProbeAnalyzer
from probes.run_manifest import compute_dataset_fingerprint, hash_indices, write_run_manifest
from probes.types import TrainedLayerProbe


class ProbeManifestAndAnalyzerTests(unittest.TestCase):
    def test_dataset_fingerprint_is_stable_and_sensitive(self) -> None:
        sample_ids = ["a", "b", "c"]
        labels = [0, 1, 0]
        fingerprint_a = compute_dataset_fingerprint(sample_ids=sample_ids, labels=labels)
        fingerprint_b = compute_dataset_fingerprint(sample_ids=sample_ids, labels=labels)
        fingerprint_c = compute_dataset_fingerprint(
            sample_ids=sample_ids, labels=[1, 1, 0]
        )
        self.assertEqual(fingerprint_a, fingerprint_b)
        self.assertNotEqual(fingerprint_a, fingerprint_c)

    def test_manifest_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            write_run_manifest(
                manifest_path=manifest_path,
                config={"probe": ProbeParams(), "sweep": SweepParams()},
                dataset_fingerprint="abc123",
                selected_key="layers_output:1",
                selection_metric="auroc",
                split_indices={"train": [0, 1], "val": [2], "test": [3]},
                split_sizes=(2, 1, 1),
                test_metrics={"auroc": 0.9},
                controls={
                    "real": {"auroc_mean": 0.9},
                    "shuffled_labels": {"auroc_mean": 0.5},
                    "random_features": {"auroc_mean": 0.5},
                },
            )
            with self.assertRaises(FileExistsError):
                write_run_manifest(
                    manifest_path=manifest_path,
                    config={"probe": ProbeParams(), "sweep": SweepParams()},
                    dataset_fingerprint="abc123",
                    selected_key="layers_output:1",
                    selection_metric="auroc",
                    split_indices={"train": [0, 1], "val": [2], "test": [3]},
                    split_sizes=(2, 1, 1),
                    test_metrics={"auroc": 0.9},
                    controls={
                        "real": {"auroc_mean": 0.9},
                        "shuffled_labels": {"auroc_mean": 0.5},
                        "random_features": {"auroc_mean": 0.5},
                    },
                )

    def test_manifest_serializes_nested_non_dataclass_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            config = {
                "artifact_dir": Path(tmp_dir) / "artifacts",
                "nested": {"values": (1, 2, 3)},
                "misc": object(),
            }
            write_run_manifest(
                manifest_path=manifest_path,
                config=config,
                dataset_fingerprint="abc123",
                selected_key="layers_output:1",
                selection_metric="auroc",
                split_indices={"train": [0, 1], "val": [2], "test": [3]},
                split_sizes=(2, 1, 1),
                test_metrics={"auroc": 0.9},
                controls={
                    "real": {"auroc_mean": 0.9},
                    "shuffled_labels": {"auroc_mean": 0.5},
                    "random_features": {"auroc_mean": 0.5},
                },
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["nested"]["values"], [1, 2, 3])
            self.assertIn("artifacts", payload["config"]["artifact_dir"])
            self.assertIsInstance(payload["config"]["misc"], str)

    def test_manifest_round_trip_contains_expected_repro_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            split_indices = {"train": [4, 0, 2], "val": [1], "test": [3, 5]}
            write_run_manifest(
                manifest_path=manifest_path,
                config={"probe": ProbeParams(), "sweep": SweepParams()},
                dataset_fingerprint="fingerprint-123",
                selected_key="layers_output:1",
                selection_metric="auroc",
                split_indices=split_indices,
                split_sizes=(3, 1, 2),
                test_metrics={"auroc": 0.9},
                controls={
                    "real": {"auroc_mean": 0.9},
                    "shuffled_labels": {"auroc_mean": 0.5},
                    "random_features": {"auroc_mean": 0.5},
                },
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_fingerprint"], "fingerprint-123")
            self.assertEqual(payload["selected_key"], "layers_output:1")
            self.assertEqual(payload["split_sizes"], {"train": 3, "val": 1, "test": 2})
            self.assertEqual(
                payload["split_index_hashes"]["train"], hash_indices(split_indices["train"])
            )
            self.assertEqual(
                payload["split_index_hashes"]["val"], hash_indices(split_indices["val"])
            )
            self.assertEqual(
                payload["split_index_hashes"]["test"], hash_indices(split_indices["test"])
            )

    @patch("matplotlib.pyplot.savefig")
    def test_analyzer_uses_lowercase_auroc_metric(self, _savefig) -> None:
        probe_low = TrainedLayerProbe(
            activation_key="layers_output:0",
            trainer=None,  # type: ignore[arg-type]
            history={},
            val_metrics={"auroc": 0.6},
            direction=torch.tensor([1.0, 0.0]),
            bias=0.0,
        )
        probe_high = TrainedLayerProbe(
            activation_key="layers_output:1",
            trainer=None,  # type: ignore[arg-type]
            history={},
            val_metrics={"auroc": 0.9},
            direction=torch.tensor([0.0, 1.0]),
            bias=0.0,
        )
        analyzer = ProbeAnalyzer([probe_low, probe_high])
        best = analyzer.auroc_analysis()
        self.assertEqual(best.activation_key, "layers_output:1")

    @patch("matplotlib.pyplot.savefig")
    def test_analyzer_uses_custom_output_paths(self, savefig_mock) -> None:
        probe = TrainedLayerProbe(
            activation_key="layers_output:0",
            trainer=None,  # type: ignore[arg-type]
            history={},
            val_metrics={"auroc": 0.7},
            direction=torch.tensor([1.0, 0.0]),
            bias=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ProbeAnalyzer([probe], output_dir=tmp_dir)
            analyzer.auroc_analysis()
            analyzer.cosine_similarity_analysis(
                output_path=Path(tmp_dir) / "custom_similarity.png"
            )
        calls = [call.args[0] for call in savefig_mock.call_args_list]
        self.assertIn(Path(tmp_dir) / "probe_auroc_ranking.png", calls)
        self.assertIn(Path(tmp_dir) / "custom_similarity.png", calls)

    @patch("matplotlib.pyplot.savefig")
    def test_analyzer_can_disable_plot_writes(self, savefig_mock) -> None:
        probe = TrainedLayerProbe(
            activation_key="layers_output:0",
            trainer=None,  # type: ignore[arg-type]
            history={},
            val_metrics={"auroc": 0.7},
            direction=torch.tensor([1.0, 0.0]),
            bias=0.0,
        )
        analyzer = ProbeAnalyzer([probe], save_plots=False)
        analyzer.auroc_analysis()
        analyzer.cosine_similarity_analysis()
        savefig_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
