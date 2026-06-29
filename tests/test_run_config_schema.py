"""Tests for stage config schemas."""

from __future__ import annotations

from sonde.core.configs.extract_config import ExtractConfig
from sonde.core.configs.generate_config import GenerateConfig
from sonde.core.configs.params.model_params import ModelParams
from sonde.core.configs.probe_config import ProbeConfig


class TestStageConfigSchemas:
    def test_probe_config_loads_composed_sections(self, tmp_path):
        yaml_text = """
run_name: smoke
seed: 0
action: probe_sweep
split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
"""
        path = tmp_path / "run.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = ProbeConfig.from_yaml(path)
        assert cfg.run_name == "smoke"
        assert cfg.split.train_fraction == 0.7

    def test_extract_config_round_trip(self, tmp_path):
        cfg = ExtractConfig(
            run_name="test_extract",
            seed=42,
            model=ModelParams(model_name="test-model"),
        )
        path = tmp_path / "extract.yaml"
        cfg.to_yaml(path)
        loaded = ExtractConfig.from_yaml(path)
        assert loaded == cfg
        assert loaded.model.model_name == "test-model"

    def test_probe_config_defaults(self):
        cfg = ProbeConfig()
        assert cfg.action == "probe_sweep"
        assert cfg.seed == 0
        assert cfg.probe.epochs == 10
        assert cfg.split.train_fraction == 0.7

    def test_generate_config_defaults(self):
        cfg = GenerateConfig()
        assert cfg.action == "generate"
        assert cfg.seed == 0
        assert cfg.generation.max_new_tokens == 256

    def test_probe_config_partial_yaml_uses_defaults(self, tmp_path):
        yaml_text = """
run_name: minimal
probe:
  epochs: 5
"""
        path = tmp_path / "run.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = ProbeConfig.from_yaml(path)
        assert cfg.run_name == "minimal"
        assert cfg.probe.epochs == 5
        assert cfg.split.train_fraction == 0.7  # default preserved
