"""Config alias resolution for friendly config selection."""

from __future__ import annotations

from pathlib import Path

import yaml


def resolve_config_alias(
    config_or_alias: str,
    aliases_path: str | Path = "configs/aliases.yaml",
) -> str:
    """Resolve a config alias to a canonical path, or pass through if not an alias.

    Args:
        config_or_alias: Either an alias key or a direct config file path.
        aliases_path: Path to the aliases YAML registry.

    Returns:
        Resolved config file path.
    """
    aliases_path = Path(aliases_path)
    if aliases_path.is_file():
        with open(aliases_path, encoding="utf-8") as f:
            aliases = yaml.safe_load(f) or {}
        if config_or_alias in aliases:
            return aliases[config_or_alias]
    return config_or_alias
