"""Tests for the CLI thin wrapper."""

from __future__ import annotations

import pytest

from cli.main import build_parser, _parse_overrides, main


class TestCli:
    def test_build_parser_accepts_config(self):
        parser = build_parser()
        args = parser.parse_args(["my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_build_parser_accepts_overrides(self):
        parser = build_parser()
        args = parser.parse_args(["cfg.yaml", "-o", "seed=42", "-o", "probe.lr=0.01"])
        assert args.config == "cfg.yaml"
        assert args.override == ["seed=42", "probe.lr=0.01"]

    def test_parse_overrides(self):
        result = _parse_overrides(["probe.lr=0.01", "seed=42"])
        assert result == {"probe.lr": "0.01", "seed": "42"}

    def test_parse_overrides_ignores_malformed(self):
        result = _parse_overrides(["good=value", "bad_no_equals"])
        assert result == {"good": "value"}

    def test_main_runs_config(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: cli_test\naction: probe_sweep\n",
            encoding="utf-8",
        )
        with pytest.raises(NotImplementedError, match="probe_sweep action not yet wired"):
            main([str(config_path)])

    def test_main_no_args_exits(self):
        with pytest.raises(SystemExit):
            main([])
