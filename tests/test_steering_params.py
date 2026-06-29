"""Tests for SteeringParams."""

from __future__ import annotations

import pytest

from sonde.core.configs.params.steering_params import SteeringParams


class TestSteeringParams:
    def test_defaults(self):
        params = SteeringParams()
        assert params.enabled is False
        assert params.vector_path == ""
        assert params.vector_key == ""
        assert params.layers == []
        assert params.strength == 10.0
        assert params.mode == "project_subtract"
        assert params.normalize is True

    def test_valid_modes(self):
        for mode in ("project_subtract", "additive"):
            p = SteeringParams(mode=mode)
            assert p.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid steering mode"):
            SteeringParams(mode="invalid")

    def test_enabled_without_vector_path_raises(self):
        with pytest.raises(ValueError, match="vector_path is required"):
            SteeringParams(enabled=True, vector_path="", layers=[0])

    def test_enabled_without_layers_raises(self):
        with pytest.raises(ValueError, match="layers is required"):
            SteeringParams(enabled=True, vector_path="v.pt", layers=[])

    def test_yaml_roundtrip(self, tmp_path):
        params = SteeringParams(
            enabled=True,
            vector_path="weights.pt",
            layers=[14, 15, 16],
            strength=15.0,
            mode="additive",
        )
        path = tmp_path / "steering.yaml"
        params.to_yaml(path)
        loaded = SteeringParams.from_yaml(path)
        assert loaded.enabled is True
        assert loaded.layers == [14, 15, 16]
        assert loaded.strength == 15.0
        assert loaded.mode == "additive"
