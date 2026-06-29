"""Config alias resolution for friendly config selection."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml


def _packaged_configs_dir() -> Path:
    """Filesystem path to the packaged ``sonde/configs`` directory."""
    return Path(str(files("sonde.configs")))


def resolve_config_alias(
    config_or_alias: str,
    aliases_path: str | Path | None = None,
) -> str:
    """Resolve a config alias to a canonical path, or pass through if not an alias.

    Args:
        config_or_alias: Either an alias key or a direct config file path.
        aliases_path: Path to an aliases YAML registry. When ``None`` (the
            default), the registry packaged at ``sonde/configs/aliases.yaml`` is
            used and its values are resolved relative to the packaged configs
            directory, so ``sonde run quickstart`` works from any directory.

    Returns:
        Resolved config file path. Built-in aliases resolve to absolute,
        package-relative paths; user-supplied registries return their values
        verbatim (back-compatible behaviour).
    """
    if aliases_path is None:
        base = _packaged_configs_dir()
        registry = base / "aliases.yaml"
        if registry.is_file():
            aliases = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
            if config_or_alias in aliases:
                return str((base / aliases[config_or_alias]).resolve())
        return config_or_alias

    aliases_file = Path(aliases_path)
    if aliases_file.is_file():
        aliases = yaml.safe_load(aliases_file.read_text(encoding="utf-8")) or {}
        if config_or_alias in aliases:
            return aliases[config_or_alias]
    return config_or_alias
