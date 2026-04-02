"""Smoke tests for the experiment runner API."""
from __future__ import annotations

import pytest

from core.configs.extract_config import ExtractConfig
from core.configs.generate_config import GenerateConfig
from core.configs.probe_config import ProbeConfig
from runners.experiment_runner import load_run_config, run_experiment


class TestExperimentRunner:
    def test_load_probe_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: smoke\nseed: 0\naction: probe_sweep\n"
            "probe:\n  learning_rate: 0.01\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path)
        assert isinstance(cfg, ProbeConfig)
        assert cfg.run_name == "smoke"
        assert cfg.probe.learning_rate == 0.01

    def test_load_generate_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "gen.yaml"
        config_path.write_text(
            "run_name: gen\naction: generate\n"
            "model:\n  model_name: test-model\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path)
        assert isinstance(cfg, GenerateConfig)
        assert cfg.model.model_name == "test-model"

    def test_load_extract_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "ext.yaml"
        config_path.write_text(
            "run_name: ext\naction: extract\n"
            "model:\n  model_name: test-model\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path)
        assert isinstance(cfg, ExtractConfig)

    def test_load_config_with_overrides(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: base\naction: probe_sweep\n"
            "probe:\n  learning_rate: 0.01\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path, overrides={"probe.learning_rate": "0.001"})
        assert isinstance(cfg, ProbeConfig)
        assert cfg.probe.learning_rate == 0.001

    def test_run_experiment_probe_sweep_raises_not_implemented(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: smoke\naction: probe_sweep\n",
            encoding="utf-8",
        )
        with pytest.raises(NotImplementedError, match="probe_sweep action not yet wired"):
            run_experiment(config_path=config_path, overrides={})

    def test_run_experiment_unknown_action_raises(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: bad\naction: nonexistent\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown action"):
            run_experiment(config_path=config_path)

    def test_run_experiment_with_alias(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text("quick: " + str(tmp_path / "quick.yaml") + "\n", encoding="utf-8")
        config_file = tmp_path / "quick.yaml"
        config_file.write_text(
            "run_name: aliased\naction: probe_sweep\n",
            encoding="utf-8",
        )
        with pytest.raises(NotImplementedError, match="probe_sweep action not yet wired"):
            run_experiment(config_path="quick", aliases_path=aliases_file)

    def test_generate_action_requires_input_path(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: gen\naction: generate\n"
            "model:\n  model_name: test-model\n"
            "generation:\n  max_new_tokens: 64\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="input_path is required"):
            run_experiment(config_path=config_path)

    def test_generate_action_missing_file_raises(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: gen\naction: generate\n"
            "model:\n  model_name: test-model\n"
            "io:\n  input_path: nonexistent.jsonl\n",
            encoding="utf-8",
        )
        with pytest.raises(FileNotFoundError):
            run_experiment(config_path=config_path)
