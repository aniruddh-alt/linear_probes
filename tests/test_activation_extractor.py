from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from activation.activation_extractor import ActivationExtractor
from activation.storage import (
    load_activation_value,
    load_extraction_manifest,
    resolve_activation_key,
)
from dataset.samples import StringDataset


def _build_extractor_stub() -> ActivationExtractor:
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    layer_module = SimpleNamespace(
        mlp=SimpleNamespace(
            gate_proj=SimpleNamespace(input="gate_in", output="gate_out"),
            down_proj=SimpleNamespace(input="down_in", output="down_out"),
        )
    )
    extractor.model = SimpleNamespace(
        num_layers=6,
        attn_probs_available=True,
        layers=[layer_module for _ in range(6)],
    )
    extractor.model_name = "stub-model"
    extractor.batch_size = 2
    extractor.default_activations = ["layers_output:0"]
    return extractor


class ActivationExtractorTests(unittest.TestCase):
    def test_module_path_activation_spec_parses(self) -> None:
        extractor = _build_extractor_stub()
        spec = extractor._parse_layer_spec("module_output:layers[3].mlp.gate_proj")
        self.assertEqual(spec.kind, "module_output")
        self.assertEqual(spec.value, "layers[3].mlp.gate_proj")

    def test_module_path_activation_spec_expands(self) -> None:
        extractor = _build_extractor_stub()
        expanded = extractor._expand_requested_activations(
            ["module_input:layers[2].mlp.down_proj"]
        )
        self.assertEqual(expanded, ["module_input:layers[2].mlp.down_proj"])

    def test_dataloader_metadata_path_is_supported(self) -> None:
        loader = DataLoader(StringDataset(["a", "b"]), batch_size=2, shuffle=False)
        prompts_source, sample_ids, labels = (
            ActivationExtractor._resolve_samples_metadata(loader)
        )
        self.assertIs(prompts_source, loader)
        self.assertEqual(sample_ids, [])
        self.assertEqual(labels, [])

    def test_select_token_position_supports_2d_when_enabled(self) -> None:
        tensor = torch.tensor([[10, 11, 12], [20, 21, 22]])
        selected = ActivationExtractor._select_token_position(
            tensor, token_index=-1, kind="input_ids", allow_2d=True
        )
        self.assertTrue(torch.equal(selected, torch.tensor([12, 22])))

    def test_select_token_position_for_attention_probabilities_uses_query_axis(self) -> None:
        tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        selected = ActivationExtractor._select_token_position(
            tensor, token_index=1, kind="attention_probabilities"
        )
        self.assertEqual(tuple(selected.shape), (2, 3, 5))
        self.assertTrue(torch.equal(selected, tensor[:, :, 1, :]))

    def test_select_token_position_raises_for_ambiguous_rank4_non_attention(self) -> None:
        tensor = torch.zeros((2, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "no unambiguous sequence axis"):
            ActivationExtractor._select_token_position(
                tensor, token_index=0, kind="module_output"
            )

    def test_storage_helper_loads_sharded_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_a = root / "a.pt"
            shard_b = root / "b.pt"
            torch.save(torch.tensor([[1.0], [2.0]]), shard_a)
            torch.save(torch.tensor([[3.0]]), shard_b)
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {},
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

            loaded = load_activation_value(extraction, activation_key="layers_output:0")
            self.assertTrue(torch.equal(loaded, torch.tensor([[1.0], [2.0], [3.0]])))

    def test_storage_helper_raises_on_non_contiguous_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard = root / "a.pt"
            torch.save(torch.tensor([[1.0]]), shard)
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {},
                "storage": {
                    "mode": "sharded",
                    "shard_index": {
                        "layers_output:0": [
                            {"path": str(shard), "start": 1, "end": 2},
                        ]
                    },
                },
            }

            with self.assertRaisesRegex(ValueError, "Non-contiguous shard index"):
                load_activation_value(extraction, activation_key="layers_output:0")

    def test_storage_helper_resolves_implicit_key_rules(self) -> None:
        extraction = {
            "requested": ["layers_output:0", "layers_output:1"],
            "activations": {
                "layers_output:0": torch.tensor([[1.0]]),
                "layers_output:1": torch.tensor([[2.0]]),
            },
            "storage": {},
        }
        with self.assertRaisesRegex(ValueError, "Multiple activation keys"):
            resolve_activation_key(extraction, activation_key=None)

    def test_storage_helper_loads_manifest_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.pt"
            extraction = {
                "requested": ["layers_output:0"],
                "activations": {"layers_output:0": torch.tensor([[1.0]])},
                "storage": {},
            }
            torch.save(extraction, manifest_path)
            loaded = load_extraction_manifest(manifest_path)
            self.assertIn("activations", loaded)
            self.assertIn("layers_output:0", loaded["activations"])


if __name__ == "__main__":
    unittest.main()
