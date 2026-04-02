from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file

from dataset.collate import sequence_collate_fn
from dataset.probing_dataset import ProbingDataset


class ProbingDatasetTests(unittest.TestCase):
    def test_raises_when_feature_dims_do_not_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dimensionality"):
            ProbingDataset(
                features=[torch.ones(2, 2), torch.ones(3)],
                labels=[0, 1],
            )

    def test_from_extraction_result_variable_length_2d_creates_sequence_mode(self) -> None:
        extraction = {
            "activations": {
                "layers_output:0": [
                    torch.ones(2, 2),
                    torch.ones(3, 2),
                ]
            },
            "labels": [0, 1],
        }
        ds = ProbingDataset.from_extraction_result(
            extraction,
            activation_key="layers_output:0",
        )
        self.assertTrue(ds.sequence_mode)
        self.assertEqual(len(ds), 2)

    def test_from_extraction_path_uses_single_available_key_when_implicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.pt"
            safetensors_path = Path(tmp_dir) / "activations.safetensors"
            save_file(
                {"layers_output:0": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                str(safetensors_path),
            )
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {},
                "sample_ids": ["a", "b"],
                "labels": [1, 0],
                "storage": {
                    "mode": "safetensors",
                    "safetensors_path": str(safetensors_path),
                },
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
            safetensors_path = Path(tmp_dir) / "activations.safetensors"
            save_file(
                {
                    "layers_output:0": torch.tensor([[1.0], [2.0]]),
                    "layers_output:1": torch.tensor([[3.0], [4.0]]),
                },
                str(safetensors_path),
            )
            extraction = {
                "requested": ["layers_output:0", "layers_output:1"],
                "activations": {},
                "sample_ids": ["a", "b"],
                "labels": [0, 1],
                "storage": {
                    "mode": "safetensors",
                    "safetensors_path": str(safetensors_path),
                },
            }
            torch.save(extraction, manifest_path)

            with self.assertRaisesRegex(ValueError, "Multiple activation keys"):
                ProbingDataset.from_extraction_path(manifest_path)

    def test_from_extraction_path_loads_safetensors_activation_and_keeps_alignment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            safetensors_path = root / "activations.safetensors"
            save_file(
                {"layers_output:0": torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])},
                str(safetensors_path),
            )

            manifest_path = root / "manifest.pt"
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {},
                "sample_ids": ["id-0", "id-1", "id-2"],
                "labels": [0, 1, 0],
                "storage": {
                    "mode": "safetensors",
                    "safetensors_path": str(safetensors_path),
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
            safetensors_path = Path(tmp_dir) / "activations.safetensors"
            save_file({"layers_output:0": torch.tensor([[1.0], [2.0]])}, str(safetensors_path))
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {},
                "sample_ids": ["a", "b"],
                "labels": [0, None],
                "storage": {
                    "mode": "safetensors",
                    "safetensors_path": str(safetensors_path),
                },
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
            safetensors_path = Path(tmp_dir) / "activations.safetensors"
            save_file({"layers_output:0": torch.tensor([[1.0], [2.0]])}, str(safetensors_path))
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {},
                "sample_ids": ["a"],
                "labels": [0, 1],
                "storage": {
                    "mode": "safetensors",
                    "safetensors_path": str(safetensors_path),
                },
            }
            torch.save(extraction, manifest_path)

            with self.assertRaisesRegex(ValueError, "sample_ids length does not match"):
                ProbingDataset.from_extraction_path(
                    manifest_path,
                    activation_key="layers_output:0",
                )



class SequenceModeProbingDatasetTests(unittest.TestCase):
    def test_sequence_dataset_returns_3_tuple(self) -> None:
        """Variable-length 2D features should trigger sequence mode."""
        features = [torch.randn(5, 8), torch.randn(10, 8), torch.randn(3, 8)]
        ds = ProbingDataset(features=features, labels=[0, 1, 0])
        self.assertEqual(len(ds), 3)
        item = ds[0]
        self.assertEqual(len(item), 3)  # (features, label, mask)
        self.assertEqual(item[0].shape, (5, 8))
        self.assertTrue(torch.all(item[2] == 1))  # all-ones mask

    def test_pooled_dataset_returns_2_tuple(self) -> None:
        """Stacked 2D tensor features should still return 2-tuple."""
        features = torch.randn(4, 8)
        ds = ProbingDataset(features=features, labels=[0, 1, 0, 1])
        item = ds[0]
        self.assertEqual(len(item), 2)

    def test_uniform_length_list_stays_pooled(self) -> None:
        """List of same-shape 1D features should stay in pooled mode."""
        features = [torch.randn(8) for _ in range(4)]
        ds = ProbingDataset(features=features, labels=[0, 1, 0, 1])
        item = ds[0]
        self.assertEqual(len(item), 2)  # pooled mode, no mask

    def test_sequence_collate_pads_to_max_len(self) -> None:
        features = [torch.randn(3, 4), torch.randn(5, 4)]
        ds = ProbingDataset(features=features, labels=[0, 1])
        batch = cast(list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], [ds[0], ds[1]])
        padded_features, _labels, mask = sequence_collate_fn(batch)
        self.assertEqual(padded_features.shape, (2, 5, 4))
        self.assertEqual(mask.shape, (2, 5))
        self.assertEqual(mask[0, :3].sum().item(), 3)
        self.assertEqual(mask[0, 3:].sum().item(), 0)
        self.assertEqual(mask[1].sum().item(), 5)

    def test_sequence_from_extraction_result(self) -> None:
        """from_extraction_result should support list-of-2D activations."""
        features = [torch.randn(5, 4), torch.randn(3, 4)]
        extraction = {
            "activations": {"layers_output:0": features},
            "labels": [0, 1],
        }
        ds = ProbingDataset.from_extraction_result(
            extraction, activation_key="layers_output:0"
        )
        self.assertEqual(len(ds), 2)
        item = ds[0]
        self.assertEqual(len(item), 3)
        self.assertEqual(item[0].shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
