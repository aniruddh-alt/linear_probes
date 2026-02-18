from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dataset.probing_dataset import ProbingDataset


class ProbingDatasetTests(unittest.TestCase):
    def test_raises_when_feature_dims_do_not_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dimensionality"):
            ProbingDataset(
                features=[torch.ones(2, 2), torch.ones(3)],
                labels=[0, 1],
            )

    def test_from_extraction_result_raises_on_inconsistent_dims(self) -> None:
        extraction = {
            "activations": {
                "layers_output:0": [
                    torch.ones(2, 2),
                    torch.ones(3, 2),
                ]
            },
            "labels": [0, 1],
        }
        with self.assertRaisesRegex(ValueError, "same dimensionality"):
            ProbingDataset.from_extraction_result(
                extraction,
                activation_key="layers_output:0",
            )

    def test_from_extraction_path_uses_single_available_key_when_implicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.pt"
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {"layers_output:0": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                "sample_ids": ["a", "b"],
                "labels": [1, 0],
                "storage": {},
            }
            torch.save(extraction, manifest_path)

            dataset = ProbingDataset.from_extraction_path(manifest_path)
            self.assertEqual(len(dataset), 2)
            self.assertTrue(torch.equal(dataset[0][0], torch.tensor([1.0, 2.0])))
            self.assertEqual(int(dataset[0][1]), 1)
            self.assertTrue(torch.equal(dataset[1][0], torch.tensor([3.0, 4.0])))
            self.assertEqual(int(dataset[1][1]), 0)

    def test_from_extraction_path_requires_key_when_manifest_has_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.pt"
            extraction = {
                "requested": ["layers_output:0", "layers_output:1"],
                "activations": {
                    "layers_output:0": torch.tensor([[1.0], [2.0]]),
                    "layers_output:1": torch.tensor([[3.0], [4.0]]),
                },
                "sample_ids": ["a", "b"],
                "labels": [0, 1],
                "storage": {},
            }
            torch.save(extraction, manifest_path)

            with self.assertRaisesRegex(ValueError, "Multiple activation keys"):
                ProbingDataset.from_extraction_path(manifest_path)

    def test_from_extraction_path_loads_sharded_activation_and_keeps_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_a = root / "shard_a.pt"
            shard_b = root / "shard_b.pt"
            torch.save(torch.tensor([[1.0, 10.0], [2.0, 20.0]]), shard_a)
            torch.save(torch.tensor([[3.0, 30.0]]), shard_b)

            manifest_path = root / "manifest.pt"
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {},
                "sample_ids": ["id-0", "id-1", "id-2"],
                "labels": [0, 1, 0],
                "storage": {
                    "mode": "sharded",
                    "shard_index": {
                        "layers_output:0": [
                            {"path": str(shard_a), "start": 0, "end": 2},
                            {"path": str(shard_b), "start": 2, "end": 3},
                        ]
                    },
                },
            }
            torch.save(extraction, manifest_path)

            dataset = ProbingDataset.from_extraction_path(
                manifest_path,
                activation_key="layers_output:0",
            )
            self.assertEqual(len(dataset), 3)
            self.assertTrue(torch.equal(dataset[0][0], torch.tensor([1.0, 10.0])))
            self.assertEqual(int(dataset[0][1]), 0)
            self.assertTrue(torch.equal(dataset[1][0], torch.tensor([2.0, 20.0])))
            self.assertEqual(int(dataset[1][1]), 1)
            self.assertTrue(torch.equal(dataset[2][0], torch.tensor([3.0, 30.0])))
            self.assertEqual(int(dataset[2][1]), 0)

    def test_from_extraction_path_raises_when_extraction_labels_are_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.pt"
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {"layers_output:0": torch.tensor([[1.0], [2.0]])},
                "sample_ids": ["a", "b"],
                "labels": [0, None],
                "storage": {},
            }
            torch.save(extraction, manifest_path)

            with self.assertRaisesRegex(ValueError, "unlabeled samples"):
                ProbingDataset.from_extraction_path(
                    manifest_path,
                    activation_key="layers_output:0",
                )

    def test_from_extraction_path_raises_when_sample_ids_do_not_align(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.pt"
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {"layers_output:0": torch.tensor([[1.0], [2.0]])},
                "sample_ids": ["a"],
                "labels": [0, 1],
                "storage": {},
            }
            torch.save(extraction, manifest_path)

            with self.assertRaisesRegex(ValueError, "sample_ids length does not match"):
                ProbingDataset.from_extraction_path(
                    manifest_path,
                    activation_key="layers_output:0",
                )


if __name__ == "__main__":
    unittest.main()
