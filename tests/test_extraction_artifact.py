"""Tests for the extraction-artifact storage format (JSON manifest + safetensors)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sonde.activation.activation_extractor import ActivationExtractor
from sonde.activation.storage import (
    SCHEMA_VERSION,
    load_activation_value,
    load_extraction_manifest,
    resolve_activation_key,
    resolve_storage_paths,
    save_extraction,
)


def _rectangular_result() -> dict:
    return {
        "model": {"name": "stub", "num_layers": 2, "hidden_size": 3},
        "requested": ["layers_output:0", "layers_output:1"],
        "activations": {
            "layers_output:0": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "layers_output:1": torch.tensor([[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]),
        },
        "sample_ids": ["a", "b"],
        "labels": [1, 0],
    }


class TestManifestFormat:
    def test_manifest_is_json_not_pickle(self, tmp_path):
        save_extraction(_rectangular_result(), tmp_path / "act")
        manifest_path = tmp_path / "act_manifest.json"
        assert manifest_path.is_file()
        # Must be plain JSON (loadable without torch / pickle).
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION
        assert "sonde_version" in payload
        assert payload["sample_ids"] == ["a", "b"]
        assert payload["labels"] == [1, 0]
        # The manifest must NOT embed the tensors.
        assert payload["storage"]["safetensors_path"] == "act.safetensors"

    def test_round_trip_values(self, tmp_path):
        result = _rectangular_result()
        save_extraction(result, tmp_path / "act")
        manifest = load_extraction_manifest(tmp_path / "act_manifest.json")
        key = resolve_activation_key(manifest, "layers_output:0")
        loaded = load_activation_value(manifest, activation_key=key)
        assert torch.equal(loaded, result["activations"]["layers_output:0"])

    def test_safetensors_path_is_relative_in_manifest(self, tmp_path):
        save_extraction(_rectangular_result(), tmp_path / "act")
        payload = json.loads((tmp_path / "act_manifest.json").read_text("utf-8"))
        # relative, not absolute — so the artifact dir can be relocated
        assert not Path(payload["storage"]["safetensors_path"]).is_absolute()


class TestRelocatable:
    def test_artifact_loads_after_directory_move(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        save_extraction(_rectangular_result(), src / "act")

        moved = tmp_path / "moved"
        shutil.move(str(src), str(moved))

        manifest = load_extraction_manifest(moved / "act_manifest.json")
        loaded = load_activation_value(manifest, activation_key="layers_output:1")
        assert torch.equal(loaded, torch.tensor([[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]))


class TestOverwriteProtection:
    def test_second_write_without_overwrite_raises(self, tmp_path):
        save_extraction(_rectangular_result(), tmp_path / "act")
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            save_extraction(_rectangular_result(), tmp_path / "act")

    def test_overwrite_true_replaces(self, tmp_path):
        save_extraction(_rectangular_result(), tmp_path / "act")
        modified = _rectangular_result()
        modified["activations"]["layers_output:0"] = torch.zeros(2, 3)
        save_extraction(modified, tmp_path / "act", overwrite=True)
        manifest = load_extraction_manifest(tmp_path / "act_manifest.json")
        loaded = load_activation_value(manifest, activation_key="layers_output:0")
        assert torch.equal(loaded, torch.zeros(2, 3))


class TestRaggedRoundTrip:
    def test_variable_length_sequences_round_trip_exactly(self, tmp_path):
        seqs = [
            torch.tensor([[1.0, 2.0]]),  # (1, 2)
            torch.tensor([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),  # (3, 2)
            torch.tensor([[9.0, 10.0], [11.0, 12.0]]),  # (2, 2)
        ]
        result = {
            "model": {"name": "stub"},
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": seqs},
            "sample_ids": ["a", "b", "c"],
            "labels": [1, 0, 1],
        }
        save_extraction(result, tmp_path / "seq")
        manifest = load_extraction_manifest(tmp_path / "seq_manifest.json")
        loaded = load_activation_value(manifest, activation_key="layers_output:0")
        assert isinstance(loaded, list)
        assert len(loaded) == 3
        for original, restored in zip(seqs, loaded, strict=True):
            assert restored.shape == original.shape
            assert torch.equal(restored, original)

    def test_manifest_marks_key_ragged(self, tmp_path):
        result = {
            "model": {},
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": [torch.zeros(2, 2), torch.zeros(4, 2)]},
            "sample_ids": ["a", "b"],
            "labels": [0, 1],
        }
        save_extraction(result, tmp_path / "seq")
        payload = json.loads((tmp_path / "seq_manifest.json").read_text("utf-8"))
        assert payload["storage"]["keys"]["layers_output:0"]["ragged"] is True


class TestSavePathNoneNoCrash:
    def _stub_extractor(self, save_path, overwrite=False) -> ActivationExtractor:
        extractor = ActivationExtractor.__new__(ActivationExtractor)
        extractor.extraction_params = SimpleNamespace(  # type: ignore[attr-defined]
            save_path=save_path, overwrite=overwrite
        )
        return extractor

    def test_none_save_path_skips_persistence(self):
        extractor = self._stub_extractor(None)
        storage = extractor._persist_result(_rectangular_result())
        assert storage == {"mode": "in_memory"}

    def test_empty_save_path_skips_persistence(self):
        extractor = self._stub_extractor("")
        storage = extractor._persist_result(_rectangular_result())
        assert storage == {"mode": "in_memory"}

    def test_set_save_path_persists(self, tmp_path):
        extractor = self._stub_extractor(str(tmp_path / "act"))
        storage = extractor._persist_result(_rectangular_result())
        assert storage["mode"] == "safetensors"
        assert (tmp_path / "act_manifest.json").is_file()
        assert (tmp_path / "act.safetensors").is_file()


class TestLegacyPickleBackCompat:
    def test_reads_legacy_pt_manifest(self, tmp_path):
        # Simulate a pre-0.1 pickle manifest with inline tensors.
        legacy = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": torch.tensor([[1.0], [2.0]])},
            "storage": {},
        }
        legacy_path = tmp_path / "old_manifest.pt"
        torch.save(legacy, legacy_path)
        loaded = load_extraction_manifest(legacy_path)
        value = load_activation_value(loaded, activation_key="layers_output:0")
        assert torch.equal(value, torch.tensor([[1.0], [2.0]]))


class TestResolveStoragePaths:
    def test_strips_pt_and_json_suffix(self):
        m, s = resolve_storage_paths("artifacts/act.pt")
        assert m.name == "act_manifest.json"
        assert s.name == "act.safetensors"
        m2, s2 = resolve_storage_paths("artifacts/act")
        assert m2.name == "act_manifest.json"
        assert s2.name == "act.safetensors"
