"""Tests for IOParams config."""

from __future__ import annotations

from sonde.core.configs.params.io_params import IOParams


class TestIOParams:
    def test_defaults(self):
        p = IOParams()
        assert p.input_path == ""
        assert p.output_dir == "artifacts"

    def test_from_dict(self):
        p = IOParams.from_dict(
            {"input_path": "data/labeled.jsonl", "output_dir": "out/"}
        )
        assert p.input_path == "data/labeled.jsonl"
        assert p.output_dir == "out/"

    def test_yaml_roundtrip(self, tmp_path):
        p = IOParams(input_path="x/y.jsonl", output_dir="z/")
        path = tmp_path / "io.yaml"
        p.to_yaml(path)
        loaded = IOParams.from_yaml(path)
        assert loaded.input_path == "x/y.jsonl"
        assert loaded.output_dir == "z/"
