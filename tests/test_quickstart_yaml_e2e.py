"""End-to-end usability validation using the quickstart recipe."""

from __future__ import annotations

from pathlib import Path

from core.configs.run_config import RunConfig
from runners.experiment_runner import load_run_config, run_experiment, RunResult


QUICKSTART_PATH = Path(__file__).resolve().parent.parent / "configs" / "recipes" / "quickstart_probe.yaml"


class TestQuickstartYamlE2E:
    def test_quickstart_yaml_loads_as_valid_run_config(self):
        cfg = RunConfig.from_yaml(QUICKSTART_PATH)
        assert cfg.run_name == "quickstart_probe"
        assert cfg.action == "probe_sweep"
        assert cfg.model.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.probe.epochs == 20
        assert cfg.probe.learning_rate == 0.01
        assert cfg.split.train_fraction == 0.7

    def test_quickstart_yaml_round_trips(self, tmp_path):
        cfg = RunConfig.from_yaml(QUICKSTART_PATH)
        out_path = tmp_path / "roundtrip.yaml"
        cfg.to_yaml(out_path)
        reloaded = RunConfig.from_yaml(out_path)
        assert reloaded == cfg

    def test_quickstart_yaml_executes_probe_flow(self):
        result = run_experiment(config_path=QUICKSTART_PATH)
        assert isinstance(result, RunResult)
        assert result.summary["run_name"] == "quickstart_probe"
        assert result.summary["action"] == "probe_sweep"
        assert result.summary["model"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_quickstart_with_override(self):
        result = run_experiment(
            config_path=QUICKSTART_PATH,
            overrides={"probe.epochs": "5", "split.train_fraction": "0.8"},
        )
        assert result.summary["run_name"] == "quickstart_probe"

    def test_load_quickstart_via_alias(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text(
            f"quickstart: {QUICKSTART_PATH}\n", encoding="utf-8"
        )
        cfg = load_run_config("quickstart", aliases_path=aliases_file)
        assert cfg.run_name == "quickstart_probe"
