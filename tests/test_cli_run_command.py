"""Tests for the CLI thin wrapper."""

from __future__ import annotations

from cli.main import build_parser, _parse_overrides, main


class TestCli:
    def test_build_parser_has_run_command(self):
        parser = build_parser()
        args = parser.parse_args(["run", "-c", "my_config.yaml"])
        assert args.command == "run"
        assert args.config == "my_config.yaml"

    def test_parse_overrides(self):
        result = _parse_overrides(["probe.lr=0.01", "seed=42"])
        assert result == {"probe.lr": "0.01", "seed": "42"}

    def test_parse_overrides_ignores_malformed(self):
        result = _parse_overrides(["good=value", "bad_no_equals"])
        assert result == {"good": "value"}

    def test_main_run_command(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: cli_test\naction: probe_sweep\nmodel:\n  model_name: m\n",
            encoding="utf-8",
        )
        rc = main(["run", "-c", str(config_path)])
        assert rc == 0

    def test_main_no_command_returns_1(self):
        rc = main([])
        assert rc == 1
