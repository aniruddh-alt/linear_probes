"""End-to-end usability validation using the quickstart recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.configs.probe_config import ProbeConfig
from runners.experiment_runner import load_run_config, run_experiment

QUICKSTART_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "recipes"
    / "quickstart_probe.yaml"
)


class TestQuickstartYamlE2E:
    def test_quickstart_yaml_loads_as_valid_probe_config(self):
        cfg = ProbeConfig.from_yaml(QUICKSTART_PATH)
        assert cfg.run_name == "quickstart_probe"
        assert cfg.action == "probe_sweep"
        assert cfg.probe.epochs == 20
        assert cfg.probe.learning_rate == 0.01
        assert cfg.split.train_fraction == 0.7

    def test_quickstart_yaml_round_trips(self, tmp_path):
        cfg = ProbeConfig.from_yaml(QUICKSTART_PATH)
        out_path = tmp_path / "roundtrip.yaml"
        cfg.to_yaml(out_path)
        reloaded = ProbeConfig.from_yaml(out_path)
        assert reloaded == cfg

    def test_quickstart_yaml_executes_probe_flow(self):
        with pytest.raises(
            NotImplementedError, match="probe_sweep action not yet wired"
        ):
            run_experiment(config_path=QUICKSTART_PATH)

    def test_quickstart_with_override(self):
        with pytest.raises(
            NotImplementedError, match="probe_sweep action not yet wired"
        ):
            run_experiment(
                config_path=QUICKSTART_PATH,
                overrides={"probe.epochs": "5", "split.train_fraction": "0.7"},
            )

    def test_load_quickstart_via_alias(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text(f"quickstart: {QUICKSTART_PATH}\n", encoding="utf-8")
        cfg = load_run_config("quickstart", aliases_path=aliases_file)
        assert isinstance(cfg, ProbeConfig)
        assert cfg.run_name == "quickstart_probe"
