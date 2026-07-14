"""Tests for ExtractConfig."""

from __future__ import annotations

from sonde.core.configs.extract_config import ExtractConfig


class TestExtractConfig:
    def test_defaults(self):
        cfg = ExtractConfig()
        assert cfg.run_name == ""
        assert cfg.action == "extract"
        assert cfg.model.model_name == ""
        assert cfg.extraction.batch_size == 8
        assert cfg.extraction.token_index == -1
        assert cfg.io.input_path == ""
        assert cfg.io.output_dir == "artifacts"

    def test_from_dict(self):
        cfg = ExtractConfig.from_dict(
            {
                "run_name": "refusal_extract",
                "model": {"model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
                "extraction": {"batch_size": 4, "token_index": -1},
                "io": {
                    "input_path": "artifacts/labeled.jsonl",
                    "output_dir": "artifacts/",
                },
            }
        )
        assert cfg.model.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.extraction.batch_size == 4
        assert cfg.io.input_path == "artifacts/labeled.jsonl"

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ExtractConfig(run_name="extract_test")
        path = tmp_path / "extract.yaml"
        cfg.to_yaml(path)
        loaded = ExtractConfig.from_yaml(path)
        assert loaded.run_name == "extract_test"
        assert loaded.action == "extract"
