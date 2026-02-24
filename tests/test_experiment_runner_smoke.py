"""Smoke tests for the experiment runner API."""

from __future__ import annotations

import pytest

from runners.experiment_runner import load_run_config, run_experiment, RunResult


class TestExperimentRunner:
    def test_load_run_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: smoke\nseed: 0\naction: probe_sweep\n"
            "model:\n  model_name: Qwen/Qwen2.5-1.5B-Instruct\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path)
        assert cfg.run_name == "smoke"
        assert cfg.model.model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_load_run_config_with_overrides(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: base\nseed: 0\naction: probe_sweep\n"
            "model:\n  model_name: base-model\n"
            "probe:\n  learning_rate: 0.01\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path, overrides={"probe.learning_rate": "0.001"})
        assert cfg.probe.learning_rate == 0.001

    def test_run_experiment_returns_structured_result(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: smoke\nseed: 0\naction: probe_sweep\n"
            "model:\n  model_name: Qwen/Qwen2.5-1.5B-Instruct\n",
            encoding="utf-8",
        )
        result = run_experiment(config_path=config_path, overrides={})
        assert isinstance(result, RunResult)
        assert "run_name" in result.summary
        assert result.summary["run_name"] == "smoke"

    def test_run_experiment_unknown_action_raises(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: bad\naction: nonexistent\n"
            "model:\n  model_name: m\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown action"):
            run_experiment(config_path=config_path)

    def test_run_experiment_with_alias(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text("quick: " + str(tmp_path / "quick.yaml") + "\n", encoding="utf-8")
        config_file = tmp_path / "quick.yaml"
        config_file.write_text(
            "run_name: aliased\naction: probe_sweep\nmodel:\n  model_name: m\n",
            encoding="utf-8",
        )
        result = run_experiment(
            config_path="quick", aliases_path=aliases_file
        )
        assert result.summary["run_name"] == "aliased"
