"""Tests for RunConfig composed schema."""

from __future__ import annotations

from core.configs.run_config import RunConfig


class TestRunConfigSchema:
    def test_run_config_loads_composed_sections(self, tmp_path):
        yaml_text = """
run_name: smoke
seed: 0
action: probe_sweep
model:
  model_name: Qwen/Qwen2.5-1.5B-Instruct
split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
"""
        path = tmp_path / "run.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = RunConfig.from_yaml(path)
        assert cfg.run_name == "smoke"
        assert cfg.model.model_name.startswith("Qwen/")

    def test_run_config_round_trip(self, tmp_path):
        from core.configs.params.model_params import ModelParams
        cfg = RunConfig(
            run_name="test_run",
            seed=42,
            action="extract",
            model=ModelParams(model_name="test-model"),
        )
        path = tmp_path / "run.yaml"
        cfg.to_yaml(path)
        loaded = RunConfig.from_yaml(path)
        assert loaded == cfg
        assert loaded.model.model_name == "test-model"

    def test_run_config_defaults(self):
        cfg = RunConfig()
        assert cfg.action == "probe_sweep"
        assert cfg.seed == 0
        assert cfg.probe.epochs == 10
        assert cfg.split.train_fraction == 0.7

    def test_run_config_partial_yaml_uses_defaults(self, tmp_path):
        yaml_text = """
run_name: minimal
model:
  model_name: my-model
"""
        path = tmp_path / "run.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = RunConfig.from_yaml(path)
        assert cfg.run_name == "minimal"
        assert cfg.probe.epochs == 10  # default preserved
        assert cfg.split.train_fraction == 0.7  # default preserved
