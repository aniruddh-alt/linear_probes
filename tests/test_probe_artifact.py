"""Tests for the ProbeArtifact (the probing -> intervention contract)."""

from __future__ import annotations

import pytest
import torch

from sonde.probes.artifact import (
    ProbeArtifact,
    layer_from_activation_key,
    save_probe_artifact,
)


class TestLayerParsing:
    def test_parses_layer_index(self):
        assert layer_from_activation_key("layers_output:15") == 15
        assert layer_from_activation_key("mlps_input:0") == 0
        assert layer_from_activation_key("token_embeddings") is None


class TestRoundTrip:
    def test_direction_bias_layer_round_trip(self, tmp_path):
        direction = torch.randn(32)
        art = ProbeArtifact(
            direction=direction,
            activation_key="layers_output:7",
            bias=0.25,
            metadata={"method": "linear_probe", "probe_type": "linear"},
        )
        assert art.layer == 7  # parsed from key
        path = art.save(tmp_path / "probe")
        loaded = ProbeArtifact.load(path)
        assert torch.allclose(loaded.direction, direction.float())
        assert loaded.activation_key == "layers_output:7"
        assert loaded.layer == 7
        assert loaded.bias == pytest.approx(0.25)
        assert loaded.metadata["method"] == "linear_probe"

    def test_no_bias_round_trip(self, tmp_path):
        art = ProbeArtifact(direction=torch.randn(8), activation_key="layers_output:1")
        path = art.save(tmp_path / "p")
        loaded = ProbeArtifact.load(path)
        assert loaded.bias is None

    def test_direction_is_flattened_float(self):
        art = ProbeArtifact(
            direction=torch.randn(1, 16).double(), activation_key="layers_output:0"
        )
        assert art.direction.ndim == 1
        assert art.direction.numel() == 16
        assert art.direction.dtype == torch.float32

    def test_requires_direction(self):
        with pytest.raises(ValueError, match="requires a direction"):
            ProbeArtifact(direction=None, activation_key="layers_output:0")  # type: ignore[arg-type]


class TestOverwrite:
    def test_overwrite_false_refuses(self, tmp_path):
        art = ProbeArtifact(direction=torch.randn(4), activation_key="layers_output:0")
        art.save(tmp_path / "p")
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            art.save(tmp_path / "p", overwrite=False)

    def test_overwrite_true_default(self, tmp_path):
        art = ProbeArtifact(direction=torch.randn(4), activation_key="layers_output:0")
        art.save(tmp_path / "p")
        art.save(tmp_path / "p")  # overwrite=True by default, no raise


class TestConvenience:
    def test_save_probe_artifact_helper(self, tmp_path):
        path = save_probe_artifact(
            direction=torch.randn(16),
            activation_key="layers_output:3",
            path=tmp_path / "d",
            bias=-0.1,
        )
        loaded = ProbeArtifact.load(path)
        assert loaded.layer == 3
        assert loaded.bias == pytest.approx(-0.1)
