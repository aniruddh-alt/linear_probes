"""Tests for config alias resolution."""

from __future__ import annotations

from sonde.core.configs.aliases import resolve_config_alias


class TestConfigAliases:
    def test_resolve_config_alias_from_registry(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text(
            "quickstart: configs/recipes/quickstart_probe.yaml\n",
            encoding="utf-8",
        )
        assert (
            resolve_config_alias("quickstart", aliases_file)
            == "configs/recipes/quickstart_probe.yaml"
        )

    def test_resolve_passthrough_when_not_alias(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text("quickstart: some/path.yaml\n", encoding="utf-8")
        assert (
            resolve_config_alias("my/custom/config.yaml", aliases_file)
            == "my/custom/config.yaml"
        )

    def test_resolve_passthrough_when_no_aliases_file(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        assert resolve_config_alias("anything", missing) == "anything"

    def test_resolve_alias_empty_file(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text("", encoding="utf-8")
        assert resolve_config_alias("quickstart", aliases_file) == "quickstart"
